"""
step25_xgboost_model.py
CRE Loan Default Risk Prediction — Phase 2
NYU Stern MSBAi Capstone 2026

REQUIRES: step24_features.parquet (run step24 first)

PURPOSE:
  Full XGBoost pipeline for 6-month-ahead CMBS loan default prediction.
  Loan-level temporal train/test split — all months of a loan go to
  either train OR test, never split across both.

OUTPUTS:
  step25_model_results.txt   — all evaluation metrics (plain text)
  step25_predictions.parquet — test set predictions + risk tiers
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
PARQUET_IN   = os.path.join(BASE, "step24_features.parquet")
RESULTS_TXT  = os.path.join(BASE, "step25_model_results.txt")
PRED_PARQUET = os.path.join(BASE, "step25_predictions.parquet")

CUTOFF_DATE  = "2022-01-01"   # start here, adjust if train/test not ~80/20

# Risk tier thresholds
TIER_THRESHOLDS = [0.10, 0.30, 0.60]

def load_features():
    df = pd.read_parquet(PARQUET_IN)
    print(f"Feature matrix: {len(df):,} rows, {df.shape[1]} columns")
    return df

def train_test_split_loan_level(df, cutoff="2022-01-01"):
    """
    Temporal loan-level split.
    Loans first observed BEFORE cutoff -> train.
    Loans first observed ON/AFTER cutoff -> test.
    All months of a loan go to the same set.
    """
    loan_first = df.groupby("ac_key")["report_period"].min()

    train_loans = loan_first[loan_first < cutoff].index
    test_loans  = loan_first[loan_first >= cutoff].index

    train = df[df["ac_key"].isin(train_loans)].copy()
    test  = df[df["ac_key"].isin(test_loans)].copy()

    print("\n=== TRAIN/TEST SPLIT ===")
    print(f"  Cutoff date:                {cutoff}")
    print(f"  Train loans:                {len(train_loans):,}  ({len(train_loans)/(len(train_loans)+len(test_loans))*100:.1f}%)")
    print(f"  Test loans:                 {len(test_loans):,}  ({len(test_loans)/(len(train_loans)+len(test_loans))*100:.1f}%)")
    print(f"  Train loan-period obs:      {len(train):,}")
    print(f"  Test loan-period obs:       {len(test):,}")
    print(f"  Train positive rate:        {train['target'].mean()*100:.2f}%")
    print(f"  Test positive rate:         {test['target'].mean()*100:.2f}%")

    train_rate = train["target"].mean()
    test_rate  = test["target"].mean()
    if abs(train_rate - test_rate) > 0.05:
        print(f"  WARNING: large distribution shift (train={train_rate:.3f}, test={test_rate:.3f})")

    return train, test, train_loans, test_loans

def carve_validation(train, val_fraction=0.10):
    """Carve last val_fraction of train loans (by first obs date) as validation."""
    loan_first = train.groupby("ac_key")["report_period"].min().sort_values()
    n_val      = max(1, int(len(loan_first) * val_fraction))
    val_loans  = loan_first.iloc[-n_val:].index
    tr_loans   = loan_first.iloc[:-n_val].index

    tr  = train[train["ac_key"].isin(tr_loans)].copy()
    val = train[train["ac_key"].isin(val_loans)].copy()
    print(f"\n  Validation carved: {len(val_loans):,} loans, {len(val):,} obs (positive: {val['target'].mean()*100:.2f}%)")
    return tr, val

def get_feature_cols(df):
    exclude = {"ac_key", "loan_uid", "report_period", "year", "state", "target",
               "ym_dt", "orig_dt", "mat_dt", "ym", "last_month", "fwd_cutoff"}
    feat_cols = [c for c in df.columns if c not in exclude]
    return feat_cols

def target_encode_state(train, test, val):
    """Target-encode state using train set mean default rate."""
    state_mean = train.groupby("state")["target"].mean().rename("state_te")
    global_mean = train["target"].mean()

    for part in [train, test, val]:
        part["state_te"] = part["state"].map(state_mean).fillna(global_mean)

    return train, test, val

def impute_features(train, test, val, feat_cols):
    """Median imputation using train set statistics."""
    medians = train[feat_cols].median()
    for part in [train, test, val]:
        for c in feat_cols:
            if c in part.columns:
                part[c] = part[c].fillna(medians[c])
    return train, test, val, medians

def train_xgboost(X_tr, y_tr, X_val, y_val):
    """Train XGBoost with early stopping."""
    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    scale_pos_weight = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)
    print(f"\n  scale_pos_weight = {scale_pos_weight:.1f}  (handles class imbalance)")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model

def evaluate_model(model, X_test, y_test, feat_cols):
    """Print full evaluation metrics."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score, confusion_matrix,
        precision_recall_curve
    )

    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)

    # Default threshold (0.5)
    pred_05  = (proba >= 0.5).astype(int)
    prec_05  = precision_score(y_test, pred_05, zero_division=0)
    rec_05   = recall_score(y_test, pred_05, zero_division=0)
    f1_05    = f1_score(y_test, pred_05, zero_division=0)

    # Best threshold by F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1s     = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_i  = np.argmax(f1s)
    best_th = thresholds[best_i] if best_i < len(thresholds) else 0.5
    pred_bt = (proba >= best_th).astype(int)
    prec_bt = precision_score(y_test, pred_bt, zero_division=0)
    rec_bt  = recall_score(y_test, pred_bt, zero_division=0)
    f1_bt   = f1_score(y_test, pred_bt, zero_division=0)

    cm = confusion_matrix(y_test, pred_bt)

    lines = [
        "\n" + "=" * 55,
        "  MODEL EVALUATION (Test Set)",
        "=" * 55,
        f"  ROC-AUC:                  {roc_auc:.4f}",
        f"  PR-AUC (Avg Precision):   {pr_auc:.4f}  <- primary metric",
        f"  Precision @ 0.5:          {prec_05:.4f}",
        f"  Recall    @ 0.5:          {rec_05:.4f}",
        f"  F1        @ 0.5:          {f1_05:.4f}",
        f"  Best threshold (max F1):  {best_th:.3f}",
        f"  Precision @ best thresh:  {prec_bt:.4f}",
        f"  Recall    @ best thresh:  {rec_bt:.4f}",
        f"  F1        @ best thresh:  {f1_bt:.4f}",
        "",
        "  CONFUSION MATRIX (@ best threshold)",
        f"                   Predicted 0   Predicted 1",
        f"  Actual 0         {cm[0,0]:>11,}   {cm[0,1]:>11,}",
        f"  Actual 1         {cm[1,0]:>11,}   {cm[1,1]:>11,}",
        "=" * 55,
    ]
    print("\n".join(lines))

    # Feature importance (top 20 by gain)
    try:
        scores = model.get_booster().get_score(importance_type="gain")
        imp_df = pd.DataFrame(list(scores.items()), columns=["feature", "gain"])
        imp_df = imp_df.sort_values("gain", ascending=False).head(20)
    except Exception:
        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(20)
        imp_df = imp_df.rename(columns={"importance": "gain"})

    print("\n  TOP 20 FEATURES BY XGBoost GAIN IMPORTANCE:")
    print("  " + "-" * 50)
    for rank, (_, r) in enumerate(imp_df.iterrows(), 1):
        print(f"  {rank:>2}. {r['feature']:<35} {r['gain']:>10.1f}")

    # SHAP (if available)
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:1000])
        mean_abs    = np.abs(shap_values).mean(axis=0)
        shap_df     = pd.DataFrame({"feature": feat_cols,
                                     "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False).head(10)
        print("\n  TOP 10 FEATURES BY MEAN |SHAP| (test set sample of 1000):")
        print("  " + "-" * 50)
        for _, r in shap_df.iterrows():
            print(f"  {r['feature']:<35} {r['mean_abs_shap']:>10.4f}")
    except ImportError:
        print("\n  (SHAP not installed — skipping SHAP analysis)")

    return proba, roc_auc, pr_auc, best_th, imp_df

def assign_risk_tiers(proba):
    """Assign 4-tier risk classification."""
    tiers = []
    for p in proba:
        if p < TIER_THRESHOLDS[0]:
            tiers.append("Tier 1 — Low Risk")
        elif p < TIER_THRESHOLDS[1]:
            tiers.append("Tier 2 — Watch")
        elif p < TIER_THRESHOLDS[2]:
            tiers.append("Tier 3 — High Risk")
        else:
            tiers.append("Tier 4 — Critical")
    return tiers

def print_risk_tier_distribution(test_df, proba, y_test):
    """Print risk tier table."""
    tiers = assign_risk_tiers(proba)
    tier_df = pd.DataFrame({
        "tier":   tiers,
        "actual": y_test.values,
    })

    tier_order = ["Tier 1 — Low Risk", "Tier 2 — Watch",
                  "Tier 3 — High Risk", "Tier 4 — Critical"]
    total = len(tier_df)

    lines = [
        "\n" + "=" * 78,
        "  RISK TIER DISTRIBUTION (Test Set)",
        "=" * 78,
        f"  {'Tier':<22} | {'Count':>8} | {'% of Total':>10} | {'Actual Default Rate':>20}",
        "  " + "-" * 68,
    ]

    for t in tier_order:
        sub   = tier_df[tier_df["tier"] == t]
        n     = len(sub)
        pct   = n / total * 100 if total > 0 else 0
        dr    = sub["actual"].mean() * 100 if n > 0 else 0
        lines.append(
            f"  {t:<22} | {n:>8,} | {pct:>9.1f}% | {dr:>19.1f}%"
        )
    lines.append("=" * 78)

    # Lender actions
    lines += [
        "",
        "  RECOMMENDED LENDER ACTIONS BY TIER:",
        "  Tier 1 — Low Risk  : Routine monitoring, standard reporting cycle.",
        "  Tier 2 — Watch     : Enhanced monitoring, quarterly credit review.",
        "  Tier 3 — High Risk : Active workout planning, reserve increase,",
        "                       monthly reporting required.",
        "  Tier 4 — Critical  : Immediate intervention, escalate to special",
        "                       servicing, initiate borrower contact.",
    ]
    print("\n".join(lines))
    return tiers

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score, average_precision_score

    with open(RESULTS_TXT, "w", encoding="utf-8") as log_file:

        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, text):
                for f in self.files:
                    f.write(text)
            def flush(self):
                for f in self.files:
                    f.flush()

        sys.stdout = Tee(sys.stdout, log_file)

        print("=" * 55)
        print("  STEP 25 — XGBOOST MODELING PIPELINE")
        print(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 55)

        df = load_features()

        # Add state target-encoding column name to features
        feat_cols = get_feature_cols(df)

        # Temporal split
        train, test, tr_loans, te_loans = train_test_split_loan_level(df, CUTOFF_DATE)

        # Adjust cutoff if train/test ratio is far from 80/20
        total_loans = len(tr_loans) + len(te_loans)
        train_pct   = len(tr_loans) / total_loans * 100
        if train_pct < 70 or train_pct > 90:
            print(f"  NOTE: Train = {train_pct:.1f}% — consider adjusting CUTOFF_DATE")

        # Carve validation from train
        train_tr, train_val = carve_validation(train)

        # Target-encode state
        train_tr, test, train_val = target_encode_state(train_tr, test, train_val)

        # Update feature cols (add state_te, remove state string)
        feat_cols = [c for c in feat_cols if c != "state"]
        if "state_te" not in feat_cols and "state_te" in train_tr.columns:
            feat_cols.append("state_te")

        # Impute
        train_tr, test, train_val, medians = impute_features(train_tr, test, train_val, feat_cols)

        X_tr  = train_tr[feat_cols]
        y_tr  = train_tr["target"]
        X_val = train_val[feat_cols]
        y_val = train_val["target"]
        X_te  = test[feat_cols]
        y_te  = test["target"]

        print(f"\n  Feature count: {len(feat_cols)}")
        print(f"  X_train shape: {X_tr.shape}")
        print(f"  X_val shape:   {X_val.shape}")
        print(f"  X_test shape:  {X_te.shape}")

        # Train
        print("\nTraining XGBoost...")
        model = train_xgboost(X_tr, y_tr, X_val, y_val)

        # Evaluate
        proba, roc_auc, pr_auc, best_th, imp_df = evaluate_model(model, X_te, y_te, feat_cols)

        # Risk tiers
        tiers = print_risk_tier_distribution(test, proba, y_te)

        # Save predictions
        test["pred_proba"] = proba
        test["risk_tier"]  = tiers
        test[["ac_key", "loan_uid", "report_period", "target", "pred_proba", "risk_tier"]].to_parquet(
            PRED_PARQUET, index=False
        )
        print(f"\nPredictions saved: {PRED_PARQUET}")

        sys.stdout = sys.stdout.files[0]

    print(f"\nModel results saved: {RESULTS_TXT}")
    print("step25 complete.")
