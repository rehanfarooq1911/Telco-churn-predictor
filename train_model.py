"""Train, tune, compare and evaluate churn models; save the best pipeline.

Usage:
    python train_model.py

Outputs:
    models/churn_pipeline.pkl      - fitted sklearn Pipeline (preprocessing + classifier)
    models/model_metadata.json     - chosen model, decision threshold, test metrics
    reports/metrics.json           - full comparison table
    reports/*.png                  - ROC curve, confusion matrix, feature importance
"""
import json
import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import randint, uniform  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from churn_features import (  # noqa: E402
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, load_and_clean,
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_pipeline.pkl"
META_PATH = "models/model_metadata.json"
RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise SystemExit(
        f"Dataset not found at {DATA_PATH}.\n"
        "Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' (see README) into the data/ folder."
    )

df = load_and_clean(DATA_PATH)
X = df[FEATURE_COLUMNS]
y = (df["Churn"] == "Yes").astype(int)
print(f"Rows: {len(df)} | churn rate: {y.mean():.1%}")

# The test set is touched exactly once, at the very end.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
neg_pos_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


# ----------------------------------------------------------------------------
# 2. Candidate models (all selection happens on the TRAIN set via cross-validation)
# ----------------------------------------------------------------------------
baseline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", DummyClassifier(strategy="prior")),
]).fit(X_train, y_train)

logreg = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

xgb_search = RandomizedSearchCV(
    Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", XGBClassifier(
            scale_pos_weight=neg_pos_ratio, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ]),
    param_distributions={
        "classifier__n_estimators": randint(100, 500),
        "classifier__max_depth": randint(2, 5),
        "classifier__learning_rate": uniform(0.01, 0.09),
        "classifier__min_child_weight": randint(1, 10),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.5, 0.5),
        "classifier__reg_lambda": uniform(1, 9),
        "classifier__gamma": uniform(0, 2),
    },
    n_iter=40, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
)
print("Tuning XGBoost (40 random configurations x 5-fold CV)...")
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
print(f"  best XGBoost CV ROC-AUC: {xgb_search.best_score_:.4f}")
best_params = {k.split("__")[1]: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v))
               for k, v in xgb_search.best_params_.items()}
print(f"  best params: {best_params}")

logreg_cv_auc = float(np.mean([
    roc_auc_score(y_train.iloc[v], logreg.fit(X_train.iloc[t], y_train.iloc[t]).predict_proba(X_train.iloc[v])[:, 1])
    for t, v in cv.split(X_train, y_train)
]))
print(f"  LogisticRegression CV ROC-AUC: {logreg_cv_auc:.4f}")

candidates = {
    "XGBoost (tuned)": (xgb_best, float(xgb_search.best_score_)),
    "Logistic Regression": (logreg, logreg_cv_auc),
}
chosen_name = max(candidates, key=lambda k: candidates[k][1])
chosen_model = candidates[chosen_name][0]
print(f"Selected on cross-validated ROC-AUC: {chosen_name}")

# ----------------------------------------------------------------------------
# 3. Decision threshold: pick the F1-optimal cutoff from out-of-fold TRAIN predictions
# ----------------------------------------------------------------------------
oof = cross_val_predict(chosen_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
grid = np.arange(0.20, 0.81, 0.01)
f1_scores = [f1_score(y_train, (oof >= t).astype(int)) for t in grid]
threshold = float(round(grid[int(np.argmax(f1_scores))], 2))
print(f"Decision threshold (max OOF F1 on train): {threshold}")

chosen_model.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# 4. Final evaluation on the untouched test set
# ----------------------------------------------------------------------------
def metrics_for(model, thr):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }, proba


results = {}
results["Baseline (majority class)"], _ = metrics_for(baseline, 0.5)
results["Logistic Regression"], _ = metrics_for(
    logreg.fit(X_train, y_train), threshold if chosen_name == "Logistic Regression" else 0.5
)
results["XGBoost (tuned)"], _ = metrics_for(
    xgb_best, threshold if chosen_name == "XGBoost (tuned)" else 0.5
)
chosen_model.fit(X_train, y_train)
final_metrics, final_proba = metrics_for(chosen_model, threshold)
train_auc = roc_auc_score(y_train, chosen_model.predict_proba(X_train)[:, 1])

print("\n=== Test-set results ===")
print(pd.DataFrame({k: {m: v[m] for m in ("accuracy", "precision", "recall", "f1", "roc_auc")}
                    for k, v in results.items()}).T.to_string())
print(f"\nSelected model: {chosen_name} @ threshold {threshold}")
print(f"  train ROC-AUC {train_auc:.3f} vs test ROC-AUC {final_metrics['roc_auc']:.3f} (overfit gap {train_auc - final_metrics['roc_auc']:.3f})")
print(f"  confusion matrix: {final_metrics['confusion_matrix']}")

# ----------------------------------------------------------------------------
# 5. Save model, metadata and report figures
# ----------------------------------------------------------------------------
joblib.dump(chosen_model, MODEL_PATH)
import sklearn  # noqa: E402
import xgboost  # noqa: E402

metadata = {
    "model": chosen_name,
    "trained_with": {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
    "threshold": threshold,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "churn_rate": round(float(y.mean()), 4),
    "cv_roc_auc": round(candidates[chosen_name][1], 4),
    "train_roc_auc": round(float(train_auc), 4),
    "test_metrics": final_metrics,
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
with open("reports/metrics.json", "w") as f:
    json.dump({"test_set_comparison": results, "selected": metadata}, f, indent=2)

fig, ax = plt.subplots(figsize=(5.5, 5))
RocCurveDisplay.from_predictions(y_test, final_proba, ax=ax, name=chosen_name)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
ax.set_title("ROC curve (test set)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("reports/roc_curve.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test, (final_proba >= threshold).astype(int), display_labels=["Stay", "Churn"],
    cmap="Blues", ax=ax, colorbar=False,
)
ax.set_title(f"Confusion matrix (test set, threshold {threshold})")
fig.tight_layout()
fig.savefig("reports/confusion_matrix.png", dpi=150)
plt.close(fig)

names = chosen_model.named_steps["preprocessor"].get_feature_names_out()
clf = chosen_model.named_steps["classifier"]
importance = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
label = "|coefficient|" if hasattr(clf, "coef_") else "importance"
top = pd.Series(importance, index=[n.split("__", 1)[1] for n in names]).nlargest(12)[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top.index, top.values, color="#2563EB")
ax.set_title(f"Top 12 features ({chosen_name}, {label})")
fig.tight_layout()
fig.savefig("reports/feature_importance.png", dpi=150)
plt.close(fig)

print(f"\nSaved {MODEL_PATH}, {META_PATH} and figures in reports/")
