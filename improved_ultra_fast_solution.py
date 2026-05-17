"""
Improved Ultra-Fast Solution targeting SMAPE < 35%
Enhanced with better feature engineering, model optimization, and validation
"""

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    return np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

def extract_enhanced_features(df):
    """Extract enhanced features with better patterns"""
    print("Extracting enhanced features...")
    
    # Enhanced IPQ extraction with multiple patterns
    def extract_ipq_enhanced(text):
        if pd.isna(text):
            return 1.0
        text_lower = str(text).lower()
        
        patterns = [
            r'(?:pack of|pack|set of|set)\s*(\d+)',
            r'(\d+)\s*(?:pack|piece|count|unit)s?',
            r'(\d+)\s*(?:x|×)',
            r'(\d+)\s*(?:count|pieces|items)',
            r'(\d+)\s*(?:units?|pieces?)',
            r'(\d+)\s*(?:pcs|pieces)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                ipq = float(match.group(1))
                return min(max(ipq, 1.0), 50.0)
        return 1.0
    
    # Enhanced brand extraction
    def extract_brand_enhanced(text):
        if pd.isna(text):
            return 'Unknown'
        lines = str(text).split('\n')
        if lines:
            item_name = lines[0].replace('Item Name:', '').strip()
            words = item_name.split()
            if words:
                brand = words[0]
                brand = re.sub(r'[^\w\s]', '', brand)
                return brand[:15]
        return 'Unknown'
    
    # Enhanced volume/weight extraction
    def extract_volume_weight_enhanced(text):
        if pd.isna(text):
            return 0, 0, 0
        text_lower = str(text).lower()
        
        volume_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:fl\s*oz|ml|liter|l)',
            r'(\d+(?:\.\d+)?)\s*(?:ounce|oz)',
            r'(\d+(?:\.\d+)?)\s*(?:gallon|gal)',
        ]
        
        weight_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:lb|kg|gram|g)',
            r'(\d+(?:\.\d+)?)\s*(?:ounce|oz)',
            r'(\d+(?:\.\d+)?)\s*(?:pound|pounds)',
        ]
        
        dimension_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:inch|in|cm|mm)',
        ]
        
        volume = 0
        weight = 0
        dimension = 0
        
        for pattern in volume_patterns:
            match = re.search(pattern, text_lower)
            if match:
                volume = float(match.group(1))
                break
        
        for pattern in weight_patterns:
            match = re.search(pattern, text_lower)
            if match:
                weight = float(match.group(1))
                break
        
        for pattern in dimension_patterns:
            match = re.search(pattern, text_lower)
            if match:
                dimension = float(match.group(1))
                break
        
        return volume, weight, dimension
    
    # Enhanced price indicators
    def extract_price_indicators_enhanced(text):
        if pd.isna(text):
            return 0, 0, 0, 0, 0
        text_lower = str(text).lower()
        
        premium_words = ['premium', 'luxury', 'deluxe', 'professional', 'gourmet', 'organic', 'natural', 'authentic']
        premium_count = sum(1 for word in premium_words if word in text_lower)
        
        size_words = ['large', 'big', 'jumbo', 'family', 'bulk', 'mega', 'giant', 'huge']
        size_count = sum(1 for word in size_words if word in text_lower)
        
        quality_words = ['high quality', 'best', 'top', 'superior', 'excellent', 'premium quality']
        quality_count = sum(1 for word in quality_words if word in text_lower)
        
        bullet_count = text_lower.count('bullet point')
        
        price_context = ['sale', 'discount', 'off', 'save', 'deal', 'special', 'limited', 'exclusive']
        price_context_count = sum(1 for word in price_context if word in text_lower)
        
        return premium_count, size_count, quality_count, bullet_count, price_context_count
    
    # Apply enhanced feature extraction
    df['ipq'] = df['catalog_content'].apply(extract_ipq_enhanced)
    df['brand'] = df['catalog_content'].apply(extract_brand_enhanced)
    
    volume_weight_dim = df['catalog_content'].apply(extract_volume_weight_enhanced)
    df['volume'] = [vwd[0] for vwd in volume_weight_dim]
    df['weight'] = [vwd[1] for vwd in volume_weight_dim]
    df['dimension'] = [vwd[2] for vwd in volume_weight_dim]
    
    # Enhanced text features
    df['text_length'] = df['catalog_content'].str.len()
    df['word_count'] = df['catalog_content'].str.split().str.len()
    df['sentence_count'] = df['catalog_content'].str.count(r'[.!?]+')
    df['paragraph_count'] = df['catalog_content'].str.count('\n\n')
    df['avg_word_length'] = df['text_length'] / df['word_count'].replace(0, 1)  # Avoid division by zero
    
    # Enhanced price indicators
    price_indicators = df['catalog_content'].apply(extract_price_indicators_enhanced)
    df['premium_count'] = [pi[0] for pi in price_indicators]
    df['size_count'] = [pi[1] for pi in price_indicators]
    df['quality_count'] = [pi[2] for pi in price_indicators]
    df['bullet_count'] = [pi[3] for pi in price_indicators]
    df['price_context_count'] = [pi[4] for pi in price_indicators]
    
    # Add missing brand_count feature
    df['brand_count'] = df['catalog_content'].str.lower().str.count('brand') + df['catalog_content'].str.lower().str.count('name') + df['catalog_content'].str.lower().str.count('label')
    
    # Enhanced structural features
    df['has_description'] = df['catalog_content'].str.contains('Product Description', case=False).astype(int)
    df['has_value'] = df['catalog_content'].str.contains('Value:', case=False).astype(int)
    df['has_unit'] = df['catalog_content'].str.contains('Unit:', case=False).astype(int)
    df['has_bullet_points'] = df['catalog_content'].str.contains('Bullet Point', case=False).astype(int)
    df['has_item_name'] = df['catalog_content'].str.contains('Item Name:', case=False).astype(int)
    
    # Text complexity features
    df['exclamation_count'] = df['catalog_content'].str.count('!')
    df['question_count'] = df['catalog_content'].str.count(r'\?')  # Fixed regex
    df['uppercase_ratio'] = df['catalog_content'].str.count(r'[A-Z]') / df['text_length'].replace(0, 1)  # Avoid division by zero
    df['digit_ratio'] = df['catalog_content'].str.count(r'\d') / df['text_length'].replace(0, 1)  # Avoid division by zero
    
    # Enhanced feature interactions
    df['ipq_volume'] = df['ipq'] * df['volume']
    df['ipq_weight'] = df['ipq'] * df['weight']
    df['text_complexity'] = df['text_length'] * df['word_count']
    df['premium_size_interaction'] = df['premium_count'] * df['size_count']
    df['quality_brand_interaction'] = df['quality_count'] * df['brand_count']
    
    df.fillna(0, inplace=True)
    return df

def prepare_enhanced_features(train_df, test_df):
    """Prepare enhanced features with better encoding"""
    print("Preparing enhanced features...")
    
    # Enhanced brand encoding
    brand_stats = train_df.groupby('brand').agg({
        'price': ['mean', 'median', 'std', 'count', 'min', 'max']
    }).round(4)
    
    brand_stats.columns = ['brand_mean', 'brand_median', 'brand_std', 'brand_count', 'brand_min', 'brand_max']
    brand_stats = brand_stats.reset_index()
    
    # Filter reliable brands
    reliable_brands = brand_stats[brand_stats['brand_count'] >= 1]
    
    # Create brand encodings
    for col in ['brand_mean', 'brand_median', 'brand_std', 'brand_count', 'brand_min', 'brand_max']:
        train_df[col] = train_df['brand'].map(reliable_brands.set_index('brand')[col])
        test_df[col] = test_df['brand'].map(reliable_brands.set_index('brand')[col])  # Fixed: use test_df instead of train_df
    
    # Fill missing brand statistics
    global_stats = {
        'brand_mean': train_df['price'].mean(),
        'brand_median': train_df['price'].median(),
        'brand_std': train_df['price'].std(),
        'brand_count': 1,
        'brand_min': train_df['price'].min(),
        'brand_max': train_df['price'].max()
    }
    
    for col, default_val in global_stats.items():
        train_df[col] = train_df[col].fillna(default_val)
        test_df[col] = test_df[col].fillna(default_val)
    
    # Create multiple text vectorizations
    tfidf_vectorizer = TfidfVectorizer(
        max_features=800,  # Increased from 300
        ngram_range=(1, 3),  # Increased from (1, 2)
        stop_words='english',
        min_df=2,
        max_df=0.9,
        lowercase=True
    )
    
    count_vectorizer = CountVectorizer(
        max_features=400,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=3,
        max_df=0.95,
        lowercase=True
    )
    
    # Fit and transform
    train_tfidf = tfidf_vectorizer.fit_transform(train_df['catalog_content'])
    test_tfidf = tfidf_vectorizer.transform(test_df['catalog_content'])
    
    train_count = count_vectorizer.fit_transform(train_df['catalog_content'])
    test_count = count_vectorizer.transform(test_df['catalog_content'])
    
    # Combine features
    basic_features = [
        'ipq', 'volume', 'weight', 'dimension',
        'brand_mean', 'brand_median', 'brand_std', 'brand_count', 'brand_min', 'brand_max',
        'text_length', 'word_count', 'sentence_count', 'paragraph_count', 'avg_word_length',
        'premium_count', 'size_count', 'quality_count', 'bullet_count', 'price_context_count',
        'has_description', 'has_value', 'has_unit', 'has_bullet_points', 'has_item_name',
        'exclamation_count', 'question_count', 'uppercase_ratio', 'digit_ratio',
        'ipq_volume', 'ipq_weight', 'text_complexity', 'premium_size_interaction', 'quality_brand_interaction'
    ]
    
    X_train_basic = train_df[basic_features].values
    X_test_basic = test_df[basic_features].values
    
    X_train_tfidf = train_tfidf.toarray()
    X_test_tfidf = test_tfidf.toarray()
    
    X_train_count = train_count.toarray()
    X_test_count = test_count.toarray()
    
    X_train = np.hstack([X_train_basic, X_train_tfidf, X_train_count])
    X_test = np.hstack([X_test_basic, X_test_tfidf, X_test_count])
    
    print(f"Enhanced features: {X_train.shape}")
    return X_train, X_test

def train_enhanced_models(X_train, y_train):
    """Train enhanced ensemble models"""
    print("Training enhanced models...")
    
    # Use log transformation
    y_log = np.log1p(y_train)
    
    # Enhanced XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,  # Increased
        max_depth=8,  # Increased
        learning_rate=0.04,  # Optimized
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # Optimized
        reg_lambda=0.1,  # Optimized
        random_state=42,
        n_jobs=-1
    )
    
    # Enhanced LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,  # Increased
        max_depth=8,  # Increased
        learning_rate=0.04,  # Optimized
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # Optimized
        reg_lambda=0.1,  # Optimized
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Additional models for ensemble
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    ridge_model = Ridge(alpha=1.0, random_state=42)
    
    # Train models
    models = {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'rf': rf_model,
        'ridge': ridge_model
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'ridge':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_log)
            models[name] = (model, scaler)
        else:
            model.fit(X_train, y_log)
    
    return models

def create_enhanced_ensemble(models, X_test):
    """Create enhanced ensemble predictions"""
    print("Creating enhanced ensemble...")
    
    # Get predictions
    predictions = {}
    for name, model in models.items():
        if name == 'ridge':
            model, scaler = model
            pred_log = model.predict(scaler.transform(X_test))
        else:
            pred_log = model.predict(X_test)
        
        pred = np.expm1(pred_log)
        pred = np.maximum(pred, 0.01)
        predictions[name] = pred
    
    # Optimized weights
    weights = {
        'xgb': 0.4,
        'lgb': 0.3,
        'rf': 0.2,
        'ridge': 0.1
    }
    
    final_pred = np.zeros(len(X_test))
    for name, pred in predictions.items():
        final_pred += weights[name] * pred
    
    return final_pred

# =====================
# New optimized pipeline
# =====================

def _get_base_models():
    """Define base learners mirroring the existing configuration."""
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
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
        verbose=-1
    )

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    ridge_model = Ridge(alpha=1.0, random_state=42)

    return {'xgb': xgb_model, 'lgb': lgb_model, 'rf': rf_model, 'ridge': ridge_model}

def _fit_models(X_train, y_log):
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

def _predict_models(models, X):
    predictions = {}
    for name, model in models.items():
        if name == 'ridge':
            est, scaler = model
            pred_log = est.predict(scaler.transform(X))
        else:
            pred_log = model.predict(X)
        pred = np.expm1(pred_log)
        pred = np.maximum(pred, 0.01)
        predictions[name] = pred
    return predictions

def optimize_blend_weights(oof_preds, y_true, step=0.1):
    """Grid search non-negative weights that sum to 1 to minimize SMAPE."""
    keys = list(oof_preds.keys())
    n = len(keys)
    best = (float('inf'), None)

    def recurse(idx, remaining, current):
        nonlocal best
        if idx == n - 1:
            w = np.array(current + [remaining])
            blend = np.zeros_like(y_true, dtype=float)
            for k, wi in zip(keys, w):
                blend += wi * oof_preds[k]
            score = calculate_smape(y_true, blend)
            if score < best[0]:
                best = (score, w)
            return
        w_i = 0.0
        while w_i <= remaining + 1e-12:
            recurse(idx + 1, remaining - w_i, current + [w_i])
            w_i += step

    recurse(0, 1.0, [])
    weights = {k: float(wi) for k, wi in zip(keys, best[1])}
    return weights, best[0]

def cross_validated_oof_and_test(X, y, X_test, n_splits=5, random_state=42):
    """Create OOF predictions per model and averaged test predictions across folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    keys = list(_get_base_models().keys())
    oof = {k: np.zeros(len(X)) for k in keys}
    test_accum = {k: np.zeros(len(X_test)) for k in keys}

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        models = _fit_models(X_tr, np.log1p(y_tr))
        va_preds = _predict_models(models, X_va)
        for k in keys:
            oof[k][va_idx] = va_preds[k]
        te_preds = _predict_models(models, X_test)
        for k in keys:
            test_accum[k] += te_preds[k] / n_splits

    return oof, test_accum

def clip_to_train_percentiles(pred, y_train, low=1.0, high=99.0):
    lo, hi = np.percentile(y_train, [low, high])
    return np.clip(pred, lo, hi)

def calibrate_by_prediction_bins(y_true, y_pred, n_bins=20, min_count=50):
    """Learn multiplicative calibration per prediction bin using OOF.
    Returns bin_edges and per-bin correction factors.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Define bins on predictions
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(y_pred, quantiles))
    # Assign bins
    bin_idx = np.digitize(y_pred, bin_edges[1:-1], right=True)
    factors = np.ones(len(bin_edges) - 1)
    for b in range(len(factors)):
        mask = bin_idx == b
        if mask.sum() >= min_count:
            med_true = np.median(y_true[mask])
            med_pred = np.median(y_pred[mask])
            if med_pred > 0:
                factors[b] = float(med_true / med_pred)
    # Clip factors to avoid extreme scaling
    factors = np.clip(factors, 0.5, 2.0)
    return bin_edges, factors

def apply_calibration(pred, bin_edges, factors):
    idx = np.digitize(pred, bin_edges[1:-1], right=True)
    return pred * factors[np.clip(idx, 0, len(factors) - 1)]

def main():
    """Main function"""
    print("="*70)
    print("IMPROVED ULTRA-FAST SOLUTION - TARGET: SMAPE -12% to -15%")
    print("="*70)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Extract enhanced features
    train_df = extract_enhanced_features(train_df)
    test_df = extract_enhanced_features(test_df)
    
    print(f"Enhanced features extracted:")
    print(f"  IPQ range: {test_df['ipq'].min()} - {test_df['ipq'].max()}")
    print(f"  Unique brands: {test_df['brand'].nunique()}")
    
    # Prepare enhanced features
    X_train, X_test = prepare_enhanced_features(train_df, test_df)
    y_train = train_df['price'].values.astype(float)
    
    # Direct price path: OOF + Test predictions per model
    print("Building OOF and test predictions (direct price)...")
    oof_direct, test_direct = cross_validated_oof_and_test(X_train, y_train, X_test, n_splits=5, random_state=42)
    weights_direct, cv_smape_direct = optimize_blend_weights(oof_direct, y_train, step=0.1)
    print(f"Direct path weights: {weights_direct} | OOF SMAPE: {cv_smape_direct:.2f}%")
    blend_direct_test = np.zeros(len(X_test))
    for k, w in weights_direct.items():
        blend_direct_test += w * test_direct[k]
    oof_blend_direct = np.zeros(len(y_train))
    for k, w in weights_direct.items():
        oof_blend_direct += w * oof_direct[k]
    
    # Unit price path (price per IPQ)
    print("Building OOF and test predictions (unit price)...")
    ipq_train = np.maximum(train_df['ipq'].values.astype(float), 1.0)
    ipq_test = np.maximum(test_df['ipq'].values.astype(float), 1.0)
    y_unit = y_train / ipq_train
    oof_unit, test_unit = cross_validated_oof_and_test(X_train, y_unit, X_test, n_splits=5, random_state=42)
    weights_unit, cv_smape_unit = optimize_blend_weights(oof_unit, y_unit, step=0.1)
    print(f"Unit path weights: {weights_unit} | OOF SMAPE (unit): {cv_smape_unit:.2f}%")
    blend_unit_test_to_price = np.zeros(len(X_test))
    for k, w in weights_unit.items():
        blend_unit_test_to_price += w * test_unit[k]
    blend_unit_test_to_price *= ipq_test
    oof_blend_unit_to_price = np.zeros(len(y_train))
    for k, w in weights_unit.items():
        oof_blend_unit_to_price += w * oof_unit[k]
    oof_blend_unit_to_price *= ipq_train
    
    # Mix the two paths via grid on OOF
    print("Tuning mixture between direct and unit-price paths...")
    best_lambda = 0.5
    best_cv_smape = float('inf')
    for lam in np.linspace(0.0, 1.0, 11):
        oof_mix = lam * oof_blend_direct + (1 - lam) * oof_blend_unit_to_price
        s = calculate_smape(y_train, oof_mix)
        if s < best_cv_smape:
            best_cv_smape = s
            best_lambda = float(lam)
    print(f"Best lambda (direct weight): {best_lambda:.2f} | OOF SMAPE: {best_cv_smape:.2f}%")
    test_mix = best_lambda * blend_direct_test + (1 - best_lambda) * blend_unit_test_to_price
    oof_mix = best_lambda * oof_blend_direct + (1 - best_lambda) * oof_blend_unit_to_price
    
    # Percentile clipping to mitigate extremes
    test_pred = clip_to_train_percentiles(test_mix, y_train, 1.0, 99.0)
    oof_clipped = clip_to_train_percentiles(oof_mix, y_train, 1.0, 99.0)
    cv_smape_clipped = calculate_smape(y_train, oof_clipped)
    print(f"After clipping OOF SMAPE: {cv_smape_clipped:.2f}%")
    
    # Bin-based post-processing: multiplicative calibration by prediction bins
    print("Learning calibration by prediction bins...")
    bin_edges, factors = calibrate_by_prediction_bins(y_train, oof_clipped, n_bins=20, min_count=50)
    test_calibrated = apply_calibration(test_pred, bin_edges, factors)
    oof_calibrated = apply_calibration(oof_clipped, bin_edges, factors)
    cv_smape_calibrated = calculate_smape(y_train, oof_calibrated)
    print(f"After calibration OOF SMAPE: {cv_smape_calibrated:.2f}%")
    
    # Final predictions
    final_pred = test_calibrated
    cv_smape = cv_smape_calibrated
    
    # Create submission
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_pred
    })
    submission_df.to_csv('test_out_improved.csv', index=False)
    
    print(f"\nPredictions saved to test_out_improved.csv")
    print(f"Count: {len(final_pred)} | Min: ${final_pred.min():.2f} | Max: ${final_pred.max():.2f} | Mean: ${final_pred.mean():.2f}")
    print("="*70)
    print(f"Estimated OOF SMAPE (post-processed): {cv_smape:.2f}%")
    print("="*70)
    
    return submission_df, cv_smape

if __name__ == "__main__":
    submission, smape = main()
