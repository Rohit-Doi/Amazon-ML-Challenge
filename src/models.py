import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor
import optuna
from functools import partial
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        epsilon: Small constant to avoid division by zero
        
    Returns:
        SMAPE score (lower is better)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Clip predictions to avoid extreme values
    y_pred = np.clip(y_pred, 0, None)
    
    # Handle zero values in y_true to avoid division by zero
    denominator = (np.abs(y_true) + np.abs(y_pred) + epsilon)
    smape_value = 200 * np.mean(np.abs(y_pred - y_true) / denominator)
    return smape_value

class ModelTrainer:
    """Handles model training, validation, and stacking."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the model trainer with configuration."""
        self.config = config or {}
        self.models = {}
        self.best_params = {}
        self.feature_importances = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        
    def train_lightgbm(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> lgb.Booster:
        """Train a LightGBM model with optional validation."""
        params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'n_jobs': -1,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X, label=y, free_raw_data=False)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=10000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            early_stopping_rounds=200,
            verbose_eval=100
        )
        
        return model
    
    def train_xgboost(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> xgb.Booster:
        """Train an XGBoost model with optional validation."""
        params = params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'random_state': 42
        }
        
        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'valid'))
            early_stopping_rounds = 200
        else:
            dval = None
            early_stopping_rounds = None
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=10000,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        return model
    
    def train_catboost(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> CatBoostRegressor:
        """Train a CatBoost model with optional validation."""
        params = params or {
            'loss_function': 'RMSE',
            'learning_rate': 0.05,
            'iterations': 10000,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 100,
            'verbose': 100
        }
        
        train_pool = xgb.DMatrix(X, label=y)
        valid_pool = xgb.DMatrix(X_val, label=y_val) if X_val is not None and y_val is not None else None
        
        model = CatBoostRegressor(**params)
        model.fit(
            X, y,
            eval_set=(X_val, y_val) if valid_pool is not None else None,
            early_stopping_rounds=200,
            verbose=100
        )
        
        return model
    
    def train_ridge(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        alpha: float = 1.0
    ) -> Ridge:
        """Train a Ridge regression model."""
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X, y)
        return model
    
    def cross_validate(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation for a given model."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_preds = np.zeros(len(X))
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            logger.info(f"\nTraining {model_name} - Fold {fold}/{n_splits}")
            
            if model_name == 'lightgbm':
                model = self.train_lightgbm(X_train, y_train, X_val, y_val, params)
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            elif model_name == 'xgboost':
                model = self.train_xgboost(X_train, y_train, X_val, y_val, params)
                y_pred = model.predict(xgb.DMatrix(X_val))
            elif model_name == 'catboost':
                model = self.train_catboost(X_train, y_train, X_val, y_val, params)
                y_pred = model.predict(X_val)
            elif model_name == 'ridge':
                model = self.train_ridge(X_train, y_train, params.get('alpha', 1.0) if params else 1.0)
                y_pred = model.predict(X_val)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Store out-of-fold predictions
            oof_preds[val_idx] = y_pred
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            smape_val = smape(y_val, y_pred)
            
            logger.info(f"Fold {fold} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, SMAPE: {smape_val:.4f}")
            scores.append({'rmse': rmse, 'mae': mae, 'smape': smape_val})
            
            # Store the first fold model for feature importance
            if fold == 1:
                self.models[model_name] = model
        
        # Calculate overall metrics
        overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        overall_mae = mean_absolute_error(y, oof_preds)
        overall_smape = smape(y, oof_preds)
        
        logger.info(f"\n{model_name} CV Results:")
        logger.info(f"Average RMSE: {np.mean([s['rmse'] for s in scores]):.4f} ± {np.std([s['rmse'] for s in scores]):.4f}")
        logger.info(f"Average MAE: {np.mean([s['mae'] for s in scores]):.4f} ± {np.std([s['mae'] for s in scores]):.4f}")
        logger.info(f"Average SMAPE: {np.mean([s['smape'] for s in scores]):.4f} ± {np.std([s['smape'] for s in scores]):.4f}")
        
        # Store OOF predictions
        self.oof_predictions[model_name] = oof_preds
        
        return {
            'model': model_name,
            'scores': scores,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'overall_smape': overall_smape,
            'oof_predictions': oof_preds
        }
    
    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        model_names: Optional[List[str]] = None,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train an ensemble of models and optionally create a stacking ensemble."""
        if model_names is None:
            model_names = ['lightgbm', 'xgboost', 'catboost', 'ridge']
        
        results = {}
        meta_features_train = []
        
        # Train individual models
        for model_name in model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name.upper()}")
            logger.info(f"{'='*50}")
            
            result = self.cross_validate(
                model_name=model_name,
                X=X,
                y=y,
                n_splits=n_splits,
                random_state=random_state
            )
            
            results[model_name] = result
            meta_features_train.append(result['oof_predictions'].reshape(-1, 1))
            
            # Make test predictions if test data is provided
            if X_test is not None:
                if model_name == 'lightgbm':
                    test_pred = self.models[model_name].predict(X_test, num_iteration=self.models[model_name].best_iteration)
                elif model_name == 'xgboost':
                    test_pred = self.models[model_name].predict(xgb.DMatrix(X_test))
                else:  # catboost, ridge, etc.
                    test_pred = self.models[model_name].predict(X_test)
                
                self.test_predictions[model_name] = test_pred
        
        # Create stacking ensemble
        if len(model_names) > 1:
            logger.info("\nTraining stacking ensemble...")
            meta_features_train = np.hstack(meta_features_train)
            
            # Use Ridge as meta-learner
            meta_learner = Ridge(alpha=1.0, random_state=random_state)
            meta_learner.fit(meta_features_train, y)
            self.models['stacking'] = meta_learner
            
            # Evaluate stacking on OOF predictions
            stack_oof_pred = meta_learner.predict(meta_features_train)
            stack_rmse = np.sqrt(mean_squared_error(y, stack_oof_pred))
            stack_mae = mean_absolute_error(y, stack_oof_pred)
            stack_smape = smape(y, stack_oof_pred)
            
            logger.info(f"Stacking Ensemble - RMSE: {stack_rmse:.4f}, MAE: {stack_mae:.4f}, SMAPE: {stack_smape:.4f}")
            
            # Make test predictions with stacking ensemble if test data is provided
            if X_test is not None and self.test_predictions:
                meta_features_test = np.column_stack([
                    self.test_predictions[model_name] for model_name in model_names
                ])
                self.test_predictions['stacking'] = meta_learner.predict(meta_features_test)
            
            results['stacking'] = {
                'model': 'stacking',
                'scores': [{'rmse': stack_rmse, 'mae': stack_mae, 'smape': stack_smape}],
                'overall_rmse': stack_rmse,
                'overall_mae': stack_mae,
                'overall_smape': stack_smape,
                'oof_predictions': stack_oof_pred
            }
        
        return results
    
    def optimize_hyperparameters(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        def objective(trial, X, y, model_name, n_splits, random_state):
            # Define search space
            if model_name == 'lightgbm':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                    'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
                    'random_state': random_state,
                    'n_jobs': -1,
                    'verbose': -1
                }
            elif model_name == 'xgboost':
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 10),
                    'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
                    'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
                    'random_state': random_state,
                    'n_jobs': -1
                }
            elif model_name == 'catboost':
                params = {
                    'loss_function': 'RMSE',
                    'iterations': 10000,
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                    'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                    'od_type': 'Iter',
                    'od_wait': 100,
                    'random_seed': random_state,
                    'verbose': 0
                }
            else:
                raise ValueError(f"Unsupported model for hyperparameter optimization: {model_name}")
            
            # Cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = []
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if model_name == 'lightgbm':
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    model = lgb.train(
                        params=params,
                        train_set=train_data,
                        valid_sets=[train_data, val_data],
                        early_stopping_rounds=100,
                        verbose_eval=False
                    )
                    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                elif model_name == 'xgboost':
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dval = xgb.DMatrix(X_val, label=y_val)
                    model = xgb.train(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=10000,
                        evals=[(dtrain, 'train'), (dval, 'val')],
                        early_stopping_rounds=100,
                        verbose_eval=False
                    )
                    y_pred = model.predict(xgb.DMatrix(X_val))
                elif model_name == 'catboost':
                    model = CatBoostRegressor(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=100,
                        verbose=0
                    )
                    y_pred = model.predict(X_val)
                
                score = smape(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, X, y, model_name, n_splits, random_state),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        logger.info(f"Best parameters for {model_name}:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        
        self.best_params[model_name] = study.best_params
        return study.best_params
