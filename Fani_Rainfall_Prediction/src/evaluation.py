import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# ============================================================
# Continuous Regression Metrics
# ============================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def bias(y_true, y_pred):
    # bias = mean(prediction - observation)
    return float(np.mean(y_pred - y_true))

def nse(y_true, y_pred):
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return float(1 - np.sum((y_true - y_pred) ** 2) / denom)

def kge(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0 or np.mean(y_true) == 0:
        return np.nan
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

def pearson_r(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])

# ============================================================
# Rain Skill Scores (Categorical)
# ============================================================
def skill_scores(y_true, y_pred, threshold):
    obs = y_true >= threshold
    fcst = y_pred >= threshold
    
    hits = int(np.sum(obs & fcst))
    misses = int(np.sum(obs & ~fcst))
    false_alarms = int(np.sum(~obs & fcst))
    correct_neg = int(np.sum(~obs & ~fcst))
    total = hits + misses + false_alarms + correct_neg
    
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    
    hits_random = ((hits + misses) * (hits + false_alarms)) / total if total > 0 else 0
    ets_denom = (hits + misses + false_alarms - hits_random)
    ets = (hits - hits_random) / ets_denom if ets_denom > 0 else np.nan
    
    return {"POD": pod, "FAR": far, "CSI": csi, "ETS": ets}

def full_metrics(y_true, y_pred, thresholds=[1.0, 5.0, 10.0]):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    
    out = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "NSE": nse(y_true, y_pred),
        "KGE": kge(y_true, y_pred),
        "r": pearson_r(y_true, y_pred)
    }
    
    for th in thresholds:
        s = skill_scores(y_true, y_pred, th)
        out[f"CSI@{th}"] = s["CSI"]
        out[f"ETS@{th}"] = s["ETS"]
        
    return out


def extreme_metrics(y_true, y_pred, threshold=15.0):
    """Return TP/FP/FN/TN breakdown and recall/precision/F1/CSI for a single threshold.
    Checks TP+FP+FN+TN == N automatically."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    N = len(y_true)
    obs  = y_true >= threshold
    fcst = y_pred >= threshold
    TP = int(np.sum(obs  &  fcst))
    FP = int(np.sum(~obs &  fcst))
    FN = int(np.sum(obs  & ~fcst))
    TN = int(np.sum(~obs & ~fcst))
    assert TP + FP + FN + TN == N, f"Matrix sum {TP+FP+FN+TN} != N={N}"
    recall    = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    f1  = 2*recall*precision/(recall+precision) if (recall+precision) > 0 else np.nan
    csi = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else np.nan
    far = FP / (TP + FP) if (TP + FP) > 0 else np.nan
    acc = (TP + TN) / N
    support = int(np.sum(obs))
    return dict(threshold=threshold, N=N, support=support,
                TP=TP, FP=FP, FN=FN, TN=TN,
                recall=recall, precision=precision, f1=f1,
                csi=csi, far=far, accuracy=acc)


def extreme_metrics_with_ci(y_true, y_pred, threshold=15.0, n_boot=5000, seed=42):
    """Extreme metrics + 95% CI via non-parametric bootstrap on individual samples."""
    rng  = np.random.default_rng(seed)
    base = extreme_metrics(y_true, y_pred, threshold)
    N    = len(y_true)
    recalls, precs, f1s, csis = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        m   = extreme_metrics(y_true[idx], y_pred[idx], threshold)
        recalls.append(m["recall"]); precs.append(m["precision"])
        f1s.append(m["f1"]);         csis.append(m["csi"])
    def ci(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        if len(arr) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    base["recall_ci95"]    = ci(recalls)
    base["precision_ci95"] = ci(precs)
    base["f1_ci95"]        = ci(f1s)
    base["csi_ci95"]       = ci(csis)
    return base


def multi_threshold_table(y_true, y_pred,
                           thresholds=(1.0, 5.0, 10.0, 15.0, 20.0, 30.0),
                           n_boot=5000):
    """Return a DataFrame of extreme metrics + CIs across multiple thresholds."""
    rows = []
    for th in thresholds:
        m = extreme_metrics_with_ci(y_true, y_pred, threshold=th, n_boot=n_boot)
        rows.append(m)
    return pd.DataFrame(rows)


# ============================================================
# Spatial Image Metrics
# ============================================================
def fractions_skill_score(actual_2d, pred_2d, threshold, window=5):
    obs_bin = (actual_2d >= threshold).astype(float)
    fcst_bin = (pred_2d >= threshold).astype(float)
    
    obs_frac = uniform_filter(obs_bin, size=window)
    fcst_frac = uniform_filter(fcst_bin, size=window)
    
    mse = np.mean((obs_frac - fcst_frac) ** 2)
    mse_ref = np.mean(obs_frac ** 2) + np.mean(fcst_frac ** 2)
    
    return float(1 - mse / mse_ref) if mse_ref > 0 else np.nan

def spatial_image_metrics(actual_2d, pred_2d):
    data_range = max(actual_2d.max(), pred_2d.max()) - min(actual_2d.min(), pred_2d.min())
    if data_range <= 0:
        return {"SSIM": np.nan, "PSNR": np.nan}
    return {
        "SSIM": float(ssim_fn(actual_2d, pred_2d, data_range=data_range)),
        "PSNR": float(psnr_fn(actual_2d, pred_2d, data_range=data_range))
    }

# ============================================================
# Probabilistic Metrics
# ============================================================
def crps_uniform_numeric(y_true, lower, upper, n_grid=200):
    crps_vals = np.empty_like(y_true, dtype="float64")
    for i in range(len(y_true)):
        a, b, y = lower[i], upper[i], y_true[i]
        if b <= a:
            crps_vals[i] = abs(y - a)
            continue
        xs = np.linspace(min(a, y) - 1, max(b, y) + 1, n_grid)
        F = np.clip((xs - a) / (b - a), 0, 1)  # Uniform CDF
        H = (xs >= y).astype(float)            # Heaviside step
        crps_vals[i] = np.trapz((F - H) ** 2, xs)
    return crps_vals

# ============================================================
# Statistical Significance & Bootstrap
# ============================================================
def paired_block_bootstrap(y_true, pred_a, pred_b, time_ids, n_boot=1000, seed=42):
    """Causal block bootstrap for comparing two model predictions."""
    rng = np.random.default_rng(seed)
    uniq_t = np.unique(time_ids)
    diffs = np.empty(n_boot)
    
    by_time_true = {t: y_true[time_ids == t] for t in uniq_t}
    by_time_a = {t: pred_a[time_ids == t] for t in uniq_t}
    by_time_b = {t: pred_b[time_ids == t] for t in uniq_t}
    
    for b in range(n_boot):
        sample_t = rng.choice(uniq_t, size=len(uniq_t), replace=True)
        y_cat = np.concatenate([by_time_true[t] for t in sample_t])
        a_cat = np.concatenate([by_time_a[t] for t in sample_t])
        b_cat = np.concatenate([by_time_b[t] for t in sample_t])
        diffs[b] = rmse(y_cat, a_cat) - rmse(y_cat, b_cat)
        
    ci = np.percentile(diffs, [2.5, 97.5])
    p_value = float(2 * min((diffs > 0).mean(), (diffs < 0).mean()))
    return float(diffs.mean()), ci, p_value

def block_bootstrap_rmse(y_by_time, pred_by_time, time_ids, n_boot=1000, seed=42):
    """Estimate confidence intervals for a model's RMSE."""
    rng = np.random.default_rng(seed)
    scores = []
    n = len(time_ids)
    for _ in range(n_boot):
        sample_ids = rng.choice(time_ids, size=n, replace=True)
        y_cat = np.concatenate([y_by_time[t] for t in sample_ids])
        p_cat = np.concatenate([pred_by_time[t] for t in sample_ids])
        scores.append(rmse(y_cat, p_cat))
    return np.array(scores)
