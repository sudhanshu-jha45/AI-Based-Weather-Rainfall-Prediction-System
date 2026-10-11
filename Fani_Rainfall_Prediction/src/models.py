import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ============================================================
# Two-Stage Rainfall Model
# ============================================================
class TwoStageRainfallModel:
    """Two-stage model: 
    Stage 1 classifies rain vs. no-rain (thresholded).
    Stage 2 regresses the intensity of positive rain.
    """
    def __init__(self, classifier, regressor, threshold=1.0):
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold
        
    def fit(self, X, y):
        # Stage 1: Classifier (predict rain >= threshold)
        y_class = (y >= self.threshold).astype(int)
        self.classifier.fit(X, y_class)
        
        # Stage 2: Regressor (fit on positive rain samples only)
        mask = y >= self.threshold
        if mask.sum() > 0:
            self.regressor.fit(X[mask], np.log1p(y[mask]))
        else:
            self.regressor.fit(X, np.log1p(y))
            
    def predict(self, X):
        # P(rain >= threshold)
        p_rain = self.classifier.predict_proba(X)[:, 1]
        # Predicted log-intensity converted back to mm
        pred_log = self.regressor.predict(X)
        pred_amount = np.expm1(pred_log)
        pred_amount = np.clip(pred_amount, 0, None)
        
        # Expected rainfall value: P(rain) * Amount(rain)
        return p_rain * pred_amount

# ============================================================
# Ensembles
# ============================================================
def greedy_ensemble_select(val_preds, y_val, max_models=10, tol=1e-6):
    """Greedily selects ensemble members on validation set to minimize RMSE."""
    from .evaluation import rmse
    selected = []
    current_blend = None
    current_score = np.inf
    
    for _ in range(max_models):
        best_name, best_score, best_trial = None, np.inf, None
        for name, pred in val_preds.items():
            trial = pred if current_blend is None else (current_blend * len(selected) + pred) / (len(selected) + 1)
            score = rmse(y_val, trial)
            if score < best_score:
                best_name, best_score, best_trial = name, score, trial
        if selected and best_score >= current_score - tol:
            break
        selected.append(best_name)
        current_blend, current_score = best_trial, best_score
        
    # Calculate weights
    from collections import Counter
    counts = Counter(selected)
    total = sum(counts.values())
    weights = {name: count / total for name, count in counts.items()}
    return selected, weights, current_score

def compute_bma_weights(val_preds, y_val):
    """Compute Bayesian Model Averaging (BMA) weights based on validation MSE."""
    names = list(val_preds.keys())
    n = len(y_val)
    log_mse = np.array([np.log(np.mean((y_val - val_preds[k])**2) + 1e-12) for k in names])
    log_lik = -0.5 * n * log_mse
    log_lik -= log_lik.max()  # Log-sum-exp stabilization
    w = np.exp(log_lik)
    w = w / w.sum()
    return dict(zip(names, w))

# ============================================================
# Model Initializers
# ============================================================
def get_baselines(X, feature_names):
    """Compute baseline predictions.
    
    1. Zero rain
    2. Persistence (lag_1)
    3. Trend extrapolation: lag_1 + (lag_1 - lag_2)
    """
    lag1_idx = feature_names.index("lag_1")
    lag2_idx = feature_names.index("lag_2")
    
    lag_1 = X[:, lag1_idx]
    lag_2 = X[:, lag2_idx]
    
    pred_zero = np.zeros(X.shape[0])
    pred_persist = lag_1
    pred_trend = np.clip(lag_1 + (lag_1 - lag_2), 0, None)
    
    return {
        "Zero Rain": pred_zero,
        "Persistence": pred_persist,
        "Trend Extrapolation": pred_trend
    }

def get_fitted_models(X_train, y_train, config):
    """Train the primary suite of ML and Tree models on training pixel dataset."""
    print("Training ML and Tree model suite...")
    
    # 1. Random Forest (tuned)
    rf = RandomForestRegressor(
        n_estimators=200, 
        max_depth=14, 
        min_samples_leaf=3, 
        n_jobs=-1, 
        random_state=config["random_seed"]
    )
    rf.fit(X_train, np.log1p(y_train))
    
    # 2. Extra Trees
    et = ExtraTreesRegressor(
        n_estimators=300, 
        max_depth=16, 
        min_samples_leaf=3, 
        n_jobs=-1, 
        random_state=config["random_seed"]
    )
    et.fit(X_train, np.log1p(y_train))
    
    # 3. XGBoost (tuned)
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=config["random_seed"]
    )
    xgb.fit(X_train, np.log1p(y_train))
    
    # 4. LightGBM
    lgbm = LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config["random_seed"],
        n_jobs=-1,
        verbosity=-1
    )
    lgbm.fit(X_train, np.log1p(y_train))
    
    # 5. CatBoost
    catb = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=config["random_seed"],
        thread_count=-1,
        verbose=False
    )
    catb.fit(X_train, np.log1p(y_train))
    
    # 6. Two-Stage Model
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        n_jobs=-1, 
        random_state=config["random_seed"]
    )
    reg = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        n_jobs=-1, 
        random_state=config["random_seed"]
    )
    ts = TwoStageRainfallModel(clf, reg, threshold=1.0)
    ts.fit(X_train, y_train)
    
    return {
        "Random Forest": rf,
        "Extra Trees": et,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "CatBoost": catb,
        "Two-Stage Model": ts
    }
