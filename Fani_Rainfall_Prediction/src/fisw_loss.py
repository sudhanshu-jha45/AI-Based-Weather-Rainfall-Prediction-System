"""
fisw_loss.py
============
Focal Intensity Sample Weighting (FISW) loss.

The paper claims:
    w(y_i) = 1 + alpha * (y_i / y_max)^gamma
    L_FISW  = (1/N) sum_i [ w(y_i)*|y_i - y_hat_i| + lambda * max(0, y_i - y_hat_i)^2 ]

For gradient-boosted tree models (XGBoost, LightGBM, CatBoost, sklearn)
the per-sample weights are passed via `sample_weight` parameter,
turning MSE into weighted MSE.

For the asymmetric under-prediction penalty term (lambda * max(0, y - yhat)^2),
we use an XGBoost custom objective.
"""

import numpy as np


# ============================================================
# Core weight function
# ============================================================

def fisw_sample_weights(y: np.ndarray, alpha: float = 15.0, gamma: float = 2.0) -> np.ndarray:
    """Compute FISW per-sample weights.

    w(y_i) = 1 + alpha * (y_i / y_max)^gamma

    Parameters
    ----------
    y      : observed rainfall (mm/h), shape (N,)
    alpha  : intensity scaling factor (default 15)
    gamma  : power exponent (default 2)

    Returns
    -------
    weights : shape (N,), all >= 1.0
    """
    y = np.asarray(y, dtype=float)
    y_max = y.max()
    if y_max < 1e-6:
        return np.ones_like(y)
    return 1.0 + alpha * (y / y_max) ** gamma


# ============================================================
# XGBoost custom FISW objective (asymmetric penalised)
# ============================================================

def fisw_xgb_objective(y_pred: np.ndarray, dtrain,
                        alpha: float = 15.0, gamma: float = 2.0,
                        lam: float = 0.5):
    """Custom XGBoost objective implementing full FISW.

    Loss = w(y) * |y - yhat| + lambda * max(0, y - yhat)^2

    Gradient and Hessian approximation for XGBoost custom objective.
    Model is trained in log1p space; y_pred is in log1p space here.
    """
    import xgboost as xgb
    y_true = dtrain.get_label()          # log1p(y_true_raw)
    y_true_raw = np.expm1(y_true)        # back to mm/h space

    # Compute residual in original space
    y_hat_raw = np.expm1(y_pred)
    residual = y_true_raw - y_hat_raw    # positive = under-prediction

    # FISW weights
    w = fisw_sample_weights(y_true_raw, alpha=alpha, gamma=gamma)

    # Gradient (dL/d_yhat in log space via chain rule)
    # dL/d_yhat_raw = -w * sign(residual) - 2*lambda*max(0, residual)
    sign_r = np.sign(residual)
    asymmetric = 2.0 * lam * np.maximum(0.0, residual)

    dL_dyhat_raw = -w * sign_r - asymmetric

    # Chain rule: d_yhat_raw / d_yhat_log = exp(y_pred) = y_hat_raw + 1
    d_log = y_hat_raw + 1.0            # derivative of expm1
    grad = dL_dyhat_raw * d_log

    # Hessian approximation (treat as constant near current point)
    hess = (w + 2.0 * lam) * d_log ** 2

    return grad.astype(np.float32), hess.astype(np.float32)


# ============================================================
# Loss ablation runner
# ============================================================

def run_loss_ablation(X_train, y_train, X_test, y_test,
                      feature_names, config, output_dir):
    """Train the same XGBoost model under 6 different loss settings.

    Returns a DataFrame with columns:
        loss_setting | MAE | RMSE | R | Recall@15 | Precision@15 | F1@15 | CSI@15
    """
    import pandas as pd
    from pathlib import Path
    from xgboost import XGBRegressor, DMatrix, train as xgb_train
    from sklearn.metrics import mean_absolute_error
    from .evaluation import rmse as calc_rmse, pearson_r, extreme_metrics

    SEED = config.get("random_seed", 42)
    THRESHOLD = 15.0   # mm/h — the headline extreme threshold

    base_params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=SEED, verbosity=0
    )

    results = []

    # ---- helper ----
    def evaluate(y_true, y_hat, label):
        mae_val  = float(mean_absolute_error(y_true, y_hat))
        rmse_val = float(calc_rmse(y_true, y_hat))
        r_val    = float(pearson_r(y_true, y_hat))
        em       = extreme_metrics(y_true, y_hat, threshold=THRESHOLD)
        results.append({
            "loss_setting": label,
            "MAE":          round(mae_val, 4),
            "RMSE":         round(rmse_val, 4),
            "Pearson_R":    round(r_val, 4),
            f"Recall@{THRESHOLD}":    round(em["recall"], 4),
            f"Precision@{THRESHOLD}": round(em["precision"], 4),
            f"F1@{THRESHOLD}":        round(em["f1"], 4),
            f"CSI@{THRESHOLD}":       round(em["csi"], 4),
        })

    # ---- 1. MSE (baseline) ----
    m = XGBRegressor(**base_params, objective="reg:squarederror")
    m.fit(X_train, np.log1p(y_train))
    pred = np.clip(np.expm1(m.predict(X_test)), 0, None)
    evaluate(y_test, pred, "MSE")

    # ---- 2. MAE ----
    m = XGBRegressor(**base_params, objective="reg:absoluteerror")
    m.fit(X_train, np.log1p(y_train))
    pred = np.clip(np.expm1(m.predict(X_test)), 0, None)
    evaluate(y_test, pred, "MAE")

    # ---- 3. Huber ----
    m = XGBRegressor(**base_params, objective="reg:pseudohubererror")
    m.fit(X_train, np.log1p(y_train))
    pred = np.clip(np.expm1(m.predict(X_test)), 0, None)
    evaluate(y_test, pred, "Huber")

    # ---- 4. Intensity weighting only (sample_weight, no asymmetric term) ----
    sw = fisw_sample_weights(y_train, alpha=15.0, gamma=2.0)
    m = XGBRegressor(**base_params, objective="reg:squarederror")
    m.fit(X_train, np.log1p(y_train), sample_weight=sw)
    pred = np.clip(np.expm1(m.predict(X_test)), 0, None)
    evaluate(y_test, pred, "Intensity Weighting Only")

    # ---- 5. Asymmetric penalty only (MAE + lambda term, equal weights) ----
    # Approximated via Tweedie (power=1.5 → intermediate between Poisson and Gamma)
    m = XGBRegressor(**base_params, objective="reg:tweedie", tweedie_variance_power=1.5)
    m.fit(X_train, y_train)
    pred = np.clip(m.predict(X_test), 0, None)
    evaluate(y_test, pred, "Asymmetric Penalty Only (Tweedie)")

    # ---- 6. Full FISW (custom objective, XGB native training loop) ----
    import xgboost as xgb
    sw_full = fisw_sample_weights(y_train, alpha=15.0, gamma=2.0)
    dtrain = xgb.DMatrix(X_train, label=np.log1p(y_train), weight=sw_full)
    dtest  = xgb.DMatrix(X_test)

    def _obj(y_pred, dtrain):
        return fisw_xgb_objective(y_pred, dtrain, alpha=15.0, gamma=2.0, lam=0.5)

    xgb_params = dict(
        max_depth=6, eta=0.05, subsample=0.8,
        colsample_bytree=0.8, seed=SEED,
        disable_default_eval_metric=True
    )
    bst = xgb_train(xgb_params, dtrain, num_boost_round=300, obj=_obj, verbose_eval=False)
    pred_log = bst.predict(dtest)
    pred = np.clip(np.expm1(pred_log), 0, None)
    evaluate(y_test, pred, "Full FISW (custom objective)")

    df = pd.DataFrame(results)
    out_path = Path(output_dir) / "loss_ablation.csv"
    df.to_csv(out_path, index=False)
    print(f"Loss ablation saved → {out_path}")
    return df
