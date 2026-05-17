"""
Optimized Ultra-Fast Solution aiming to reduce SMAPE by ~10%

Key improvements over improved_ultra_fast_solution.py:
- CV-based non-negative blending weight optimization (sum-to-1) to minimize SMAPE
- Dual-path modeling: direct price and unit price (price per ipq), with CV-tuned mixing
- Robust percentile clipping of predictions to reduce extremes
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Reuse feature extraction utilities from the existing solution
from improved_ultra_fast_solution import (
    extract_enhanced_features,
    prepare_enhanced_features,
    calculate_smape,
)


def _get_base_models():
    """Define base learners with the same settings as the existing solution."""
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
    )

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    ridge_model = Ridge(alpha=1.0, random_state=42)

    return {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'rf': rf_model,
        'ridge': ridge_model,
    }


def _fit_models(X_train: np.ndarray, y_log: np.ndarray):
    """Fit base models; for Ridge, apply scaling. Return fitted models dict."""
    models = _get_base_models()
    fitted = {}
    for name, model in models.items():
        if name == 'ridge':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            model.fit(X_scaled, y_log)
            fitted[name] = (model, scaler)
        else:
            model.fit(X_train, y_log)
            fitted[name] = model
    return fitted


def _predict_models(models: dict, X: np.ndarray) -> dict:
    """Predict with base models dict (handling Ridge scaler). Returns dict of linear-space preds."""
    preds = {}
    for name, model in models.items():
        if name == 'ridge':
            model_est, scaler = model
            pred_log = model_est.predict(scaler.transform(X))
        else:
            pred_log = model.predict(X)
        pred = np.expm1(pred_log)
        pred = np.maximum(pred, 0.01)
        preds[name] = pred
    return preds


def optimize_blend_weights(oof_preds: dict, y_true: np.ndarray, step: float = 0.05) -> dict:
    """Grid search non-negative weights (sum to 1) minimizing SMAPE on OOF.

    Args:
        oof_preds: dict[name] -> np.ndarray of OOF predictions
        y_true: ground truth array
        step: grid step size for weights

    Returns:
        dict of weights per model key
    """
    keys = list(oof_preds.keys())
    n = len(keys)
    best_smape = float('inf')
    best_w = None

    # Simple recursive grid generator ensuring sum to 1
    def recurse(idx, remaining, current):
        nonlocal best_smape, best_w
        if idx == n - 1:
            w = current + [remaining]
            w = np.array(w)
            blended = np.zeros_like(y_true, dtype=float)
            for k, weight in zip(keys, w):
                blended += weight * oof_preds[k]
            score = calculate_smape(y_true, blended)
            if score < best_smape:
                best_smape = score
                best_w = w.copy()
            return
        for w_i in np.arange(0.0, remaining + 1e-12, step):
            recurse(idx + 1, remaining - w_i, current + [w_i])

    recurse(0, 1.0, [])
    weights = {k: float(wi) for k, wi in zip(keys, best_w)}
    return weights


def cross_validated_oof_and_test(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
):
    """Generate OOF predictions per model and averaged test predictions across folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = {k: np.zeros(len(X)) for k in _get_base_models().keys()}
    test_preds_accum = {k: np.zeros(len(X_test)) for k in _get_base_models().keys()}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Train in log-space
        y_tr_log = np.log1p(y_tr)
        models = _fit_models(X_tr, y_tr_log)

        # Predict for validation and test
        val_preds = _predict_models(models, X_val)
        for k in oof_preds:
            oof_preds[k][val_idx] = val_preds[k]

        test_fold_preds = _predict_models(models, X_test)
        for k in test_preds_accum:
            test_preds_accum[k] += test_fold_preds[k] / n_splits

    return oof_preds, test_preds_accum


def clip_to_train_percentiles(pred: np.ndarray, y_train: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(y_train, [low, high])
    return np.clip(pred, lo, hi)


def main():
    print("=" * 70)
    print("OPTIMIZED ULTRA-FAST SOLUTION - TARGET: -10% SMAPE")
    print("=" * 70)

    # Load data
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')

    # Feature extraction (reuse existing robust extractors)
    train_df = extract_enhanced_features(train_df)
    test_df = extract_enhanced_features(test_df)

    # Prepare matrices
    X_train, X_test = prepare_enhanced_features(train_df, test_df)
    y_train = train_df['price'].values.astype(float)

    # Path A: direct price modeling
    print("Generating OOF and test predictions for direct price path...")
    oof_direct, test_direct = cross_validated_oof_and_test(X_train, y_train, X_test)

    # Optimize blending weights on OOF
    print("Optimizing blend weights for direct price...")
    weights_direct = optimize_blend_weights(oof_direct, y_train, step=0.1)
    print(f"Direct price weights: {weights_direct}")

    # Blend test predictions using optimized weights
    blended_test_direct = np.zeros(len(X_test))
    for k, w in weights_direct.items():
        blended_test_direct += w * test_direct[k]

    # Path B: unit price modeling (price per ipq)
    print("Generating OOF and test predictions for unit price path...")
    ipq_train = np.maximum(train_df['ipq'].values.astype(float), 1.0)
    ipq_test = np.maximum(test_df['ipq'].values.astype(float), 1.0)
    y_unit = y_train / ipq_train

    oof_unit, test_unit = cross_validated_oof_and_test(X_train, y_unit, X_test)
    print("Optimizing blend weights for unit price...")
    weights_unit = optimize_blend_weights(oof_unit, y_unit, step=0.1)
    print(f"Unit price weights: {weights_unit}")

    blended_test_unit = np.zeros(len(X_test))
    for k, w in weights_unit.items():
        blended_test_unit += w * test_unit[k]
    blended_test_unit_to_price = blended_test_unit * ipq_test

    # Find best mix between direct and unit-price paths via CV grid on OOF
    print("Tuning mix between direct and unit-price paths...")
    oof_blend_direct = np.zeros(len(y_train))
    for k, w in weights_direct.items():
        oof_blend_direct += w * oof_direct[k]

    oof_blend_unit = np.zeros(len(y_train))
    for k, w in weights_unit.items():
        oof_blend_unit += w * oof_unit[k]
    oof_blend_unit_to_price = oof_blend_unit * ipq_train

    best_lambda = 0.5
    best_cv_smape = float('inf')
    for lam in np.linspace(0.0, 1.0, 11):
        oof_mix = lam * oof_blend_direct + (1 - lam) * oof_blend_unit_to_price
        score = calculate_smape(y_train, oof_mix)
        if score < best_cv_smape:
            best_cv_smape = score
            best_lambda = float(lam)

    print(f"Best mixing lambda (direct weight): {best_lambda:.2f}, OOF SMAPE: {best_cv_smape:.2f}%")

    # Apply mix to test predictions
    test_mix = best_lambda * blended_test_direct + (1 - best_lambda) * blended_test_unit_to_price

    # Clip to robust percentiles
    test_pred = clip_to_train_percentiles(test_mix, y_train, 1.0, 99.0)

    # Report CV
    cv_smape = best_cv_smape

    # Save submission
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_pred,
    })
    out_path = 'test_out_optimized.csv'
    submission_df.to_csv(out_path, index=False)

    print("\nOptimized predictions saved to test_out_optimized.csv")
    print(f"Count: {len(test_pred)} | Min: {test_pred.min():.2f} | Max: {test_pred.max():.2f} | Mean: {test_pred.mean():.2f}")
    print("=" * 70)
    print(f"Estimated CV SMAPE: {cv_smape:.2f}% (optimized)")
    print("=" * 70)

    return submission_df, cv_smape


if __name__ == "__main__":
    submission, smape = main()


