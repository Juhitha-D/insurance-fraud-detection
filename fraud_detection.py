"""
Insurance Fraud Detection Using Machine Learning
-------------------------------------------------
Dataset : Kaggle "Auto Insurance Claims" (insurance_claims.csv, 1000 rows)
Target  : fraud_reported (Y / N)

Usage   : python fraud_detection.py --data insurance_claims.csv --out results

Pipeline
  1. Load + clean ('?' -> 'Unknown', drop IDs / leakage-prone / useless columns)
  2. Light feature engineering (dates -> customer tenure, claim ratios)
  3. Stratified 80/20 train/test split (random_state fixed for reproducibility)
  4. Compare 5 classifiers, each with and without class weighting,
     using 5-fold stratified cross-validation on the training set
  5. Tune the decision threshold of the best model on out-of-fold predictions
  6. Final evaluation on the untouched test set
  7. Save tables + figures for the slides and report
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42

# Columns that are identifiers, free text, or empty -> no predictive meaning
DROP_COLS = ["policy_number", "insured_zip", "incident_location", "_c39",
             "policy_bind_date", "incident_date", "auto_model"]


# ----------------------------------------------------------------------------- data
def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace("?", "Unknown")

    # Feature engineering from dates: how long the policy existed before the incident
    if {"policy_bind_date", "incident_date"} <= set(df.columns):
        bind = pd.to_datetime(df["policy_bind_date"], errors="coerce")
        inc = pd.to_datetime(df["incident_date"], errors="coerce")
        df["policy_age_days"] = (inc - bind).dt.days

    # Ratios of claim components (how the total is split)
    if "total_claim_amount" in df.columns:
        tot = df["total_claim_amount"].replace(0, np.nan)
        for c in ["injury_claim", "property_claim", "vehicle_claim"]:
            if c in df.columns:
                df[f"{c}_ratio"] = (df[c] / tot).fillna(0)

    y = (df["fraud_reported"].astype(str).str.upper().str.startswith("Y")).astype(int)
    X = df.drop(columns=[c for c in DROP_COLS + ["fraud_reported"] if c in df.columns])
    # pandas reads the text "None" (e.g. authorities_contacted) as missing -> restore it
    obj = X.select_dtypes(include="object").columns
    X[obj] = X[obj].fillna("None")
    return X, y


def make_preprocessor(X):
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False), cat),
    ]), num, cat


# --------------------------------------------------------------------------- models
def get_models():
    """Each model in two variants: plain and class-weighted (imbalance handling)."""
    base = {
        "Logistic Regression": lambda cw: LogisticRegression(max_iter=2000, C=0.5, class_weight=cw),
        "Decision Tree":       lambda cw: DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                                                 class_weight=cw, random_state=SEED),
        "Random Forest":       lambda cw: RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                                 class_weight=cw, n_jobs=-1, random_state=SEED),
        "Gradient Boosting":   lambda cw: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                         max_iter=300, class_weight=cw,
                                                                         random_state=SEED),
        "SVM (RBF)":           lambda cw: SVC(C=1.0, probability=True, class_weight=cw, random_state=SEED),
    }
    models = {}
    for name, fn in base.items():
        models[name] = fn(None)
        models[name + " (balanced)"] = fn("balanced")
    return models


def scores(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "PR-AUC":    average_precision_score(y_true, y_prob),
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
    }


# ----------------------------------------------------------------------------- main
def main(data_path, out):
    os.makedirs(out, exist_ok=True)
    X, y = load_and_clean(data_path)
    print(f"Data: {X.shape[0]} rows, {X.shape[1]} features, fraud rate = {y.mean():.1%}")

    # Fig 1: class distribution
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = y.value_counts().sort_index()
    ax.bar(["Legitimate", "Fraud"], counts.values, color=["#4C78A8", "#E45756"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / len(y):.0%})", ha="center", va="bottom")
    ax.set_ylabel("Number of claims"); ax.set_title("Class distribution")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out}/fig1_class_distribution.png", dpi=200); plt.close(fig)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pre, num, cat = make_preprocessor(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # --- 5-fold CV comparison on training set (default 0.5 threshold)
    rows, oof = [], {}
    models = {"Baseline (majority)": DummyClassifier(strategy="most_frequent")}
    models.update(get_models())
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        prob = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        oof[name] = prob
        r = scores(y_tr, (prob >= 0.5).astype(int), prob); r["Model"] = name
        rows.append(r)
        print(f"  {name:35s} F1={r['F1']:.3f}  PR-AUC={r['PR-AUC']:.3f}  Recall={r['Recall']:.3f}")
    cv_df = pd.DataFrame(rows).set_index("Model").round(3)
    cv_df.to_csv(f"{out}/table1_cv_results.csv")

    # --- pick best by PR-AUC (threshold-independent, suited to imbalance)
    best = cv_df.drop(index="Baseline (majority)")["PR-AUC"].idxmax()
    print(f"\nBest model by CV PR-AUC: {best}")

    # --- threshold tuning on out-of-fold probabilities (maximise F1)
    p, r, t = precision_recall_curve(y_tr, oof[best])
    f1s = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    thr = float(t[np.argmax(f1s)])
    print(f"Tuned threshold (max OOF F1): {thr:.3f}")

    # --- Fig 2: CV PR curves for the best variant of each model family
    fig, ax = plt.subplots(figsize=(6, 4.5))
    families = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVM (RBF)"]
    for fam in families:
        name = max([fam, fam + " (balanced)"], key=lambda n: cv_df.loc[n, "PR-AUC"])
        pp, rr, _ = precision_recall_curve(y_tr, oof[name])
        ax.plot(rr, pp, label=f"{fam} (AP={cv_df.loc[name, 'PR-AUC']:.2f})")
    ax.axhline(y_tr.mean(), ls="--", c="grey", label=f"Random (AP={y_tr.mean():.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall curves (5-fold CV)")
    ax.legend(fontsize=8, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out}/fig2_pr_curves_cv.png", dpi=200); plt.close(fig)

    # --- Fig 3: F1 vs PR-AUC bar chart of all models (CV)
    plot_df = cv_df.drop(index="Baseline (majority)")[["F1", "PR-AUC", "Recall"]].sort_values("PR-AUC")
    ax = plot_df.plot.barh(figsize=(7, 5.5), color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_xlabel("Score (5-fold CV)"); ax.set_title("Model comparison"); ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{out}/fig3_model_comparison.png", dpi=200); plt.close()

    # --- Final fit on full train, evaluate once on held-out test
    final = Pipeline([("pre", pre), ("clf", get_models()[best])]).fit(X_tr, y_tr)
    prob_te = final.predict_proba(X_te)[:, 1]
    test_rows = []
    for label, t_ in [("default (0.50)", 0.5), (f"tuned ({thr:.2f})", thr)]:
        s = scores(y_te, (prob_te >= t_).astype(int), prob_te)
        s["Model"] = f"{best} – threshold {label}"; test_rows.append(s)
    # also report every model on test at 0.5 for completeness
    for name, clf in get_models().items():
        pr = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        s = scores(y_te, (pr >= 0.5).astype(int), pr); s["Model"] = name; test_rows.append(s)
    test_df = pd.DataFrame(test_rows).set_index("Model").round(3)
    test_df.to_csv(f"{out}/table2_test_results.csv")
    print("\nTest set:\n", test_df.head(2).to_string())

    # --- Fig 4: confusion matrices on test (default vs tuned threshold)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (label, t_) in zip(axes, [("threshold 0.50", 0.5), (f"tuned threshold {thr:.2f}", thr)]):
        cm = confusion_matrix(y_te, (prob_te >= t_).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(label, fontsize=10)
    fig.suptitle(f"{best} – test set"); fig.tight_layout()
    fig.savefig(f"{out}/fig4_confusion_matrices.png", dpi=200); plt.close(fig)

    # --- Fig 5: permutation importance (on raw columns, test set, scored by PR-AUC)
    pi = permutation_importance(final, X_te, y_te, scoring="average_precision",
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
    imp.round(4).to_csv(f"{out}/table3_feature_importance.csv")
    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top.index, top.values, color="#4C78A8")
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled"); ax.set_title("Top 10 features (permutation importance)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out}/fig5_feature_importance.png", dpi=200); plt.close(fig)
    print("\nTop features:\n", imp.head(10).round(4).to_string())

    # --- Fig 6: fraud rate by the two most important categorical features (insight slides)
    for top_cat in [c for c in imp.index if c in cat][:2]:
        rate = pd.DataFrame({"f": X[top_cat], "y": y}).groupby("f")["y"].agg(["mean", "size"])
        rate = rate[rate["size"] >= 10].sort_values("mean")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(rate.index.astype(str), rate["mean"], color="#E45756")
        ax.axvline(y.mean(), ls="--", c="grey", label=f"overall {y.mean():.0%}")
        ax.set_xlabel("Fraud rate"); ax.set_title(f"Fraud rate by {top_cat}"); ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{out}/fig6_fraud_rate_by_{top_cat}.png", dpi=200); plt.close(fig)

    with open(f"{out}/summary.txt", "w") as f:
        f.write(f"rows={len(X)} features={X.shape[1]} fraud_rate={y.mean():.4f}\n")
        f.write(f"best_model={best}\ntuned_threshold={thr:.4f}\n\nCV results\n{cv_df.to_string()}\n")
        f.write(f"\nTest results\n{test_df.to_string()}\n\nPermutation importance\n{imp.head(15).to_string()}\n")
    print(f"\nAll outputs saved to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="insurance_claims.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.data, a.out)
