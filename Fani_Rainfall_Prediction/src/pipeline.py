import os
import json
import time
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, KEY_LOCATIONS
from .data_loader import load_and_merge_datasets, deaccumulate_rainfall
from .data_quality import run_data_quality_checks
from .features import load_and_clean_track, build_pixel_dataset, get_feature_names
from .leakage_audit import run_leakage_audit
from .models import get_baselines, get_fitted_models, TwoStageRainfallModel, greedy_ensemble_select, compute_bma_weights
from .evaluation import full_metrics, rmse, mae, r2_score, kge, pearson_r, paired_block_bootstrap, skill_scores
from .diagnostics import (
    plot_fig01_fani_track, plot_fig02_precip_spatial, plot_fig03_temporal_evolution,
    plot_fig04_rainfall_distribution, plot_fig05_extreme_rainfall_map, plot_fig06_observed_vs_predicted,
    plot_fig07_residual_distribution, plot_fig08_model_comparison, plot_fig09_storm_relative_structure,
    plot_fig10_rain_vs_distance, plot_fig11_rain_vs_intensity, plot_fig12_rain_vs_speed,
    plot_fig13_spatial_error, plot_fig14_temporal_error, plot_fig15_extreme_performance,
    plot_fig16_uncertainty_calibration, plot_fig17_feature_importance, plot_fig18_ablation,
    plot_fig19_case_study, generate_tables
)

def run_end_to_end_pipeline():
    start_time = time.time()
    print("=" * 100)
    print("CYCLONE FANI PRECIPITATION PREDICTION SYSTEM")
    print("=" * 100)
    
    # 1. Configuration & Directories
    print("Initializing directories...")
    
    # 2. Data Loading & Harmonization
    raw_merged = load_and_merge_datasets(CONFIG["source_a_path"], CONFIG["source_b_path"])
    
    # 3. Precipitation Physics (Deaccumulation)
    rainfall, reset_mask = deaccumulate_rainfall(raw_merged)
    
    # 4. Data Quality Audit
    dq_report = run_data_quality_checks(rainfall, reset_mask, CONFIG)
    
    # 5. Storm Track Processing
    track_df = load_and_clean_track(CONFIG["track_path"])
    print(f"Loaded IMD Best Track containing {len(track_df)} records.")
    
    # Find matching time indices where track information is available
    rain_times = pd.DatetimeIndex(rainfall.time.values)
    track_start = track_df["time"].min()
    track_end = track_df["time"].max()
    
    # Restrict modelling period to overlapping timeline
    in_track_mask = (rain_times >= track_start) & (rain_times <= track_end)
    overlap_indices = np.where(in_track_mask)[0]
    
    # Exclude first SEQ_LEN=4 indices of track to allow history setup
    seq_len = CONFIG["seq_len"]
    seq_time_idx = overlap_indices[seq_len:]
    total_sequences = len(seq_time_idx)
    
    # 6. Temporal Splitting
    n_train = int(total_sequences * 0.6)
    n_val = int(total_sequences * 0.2)
    n_test = total_sequences - n_train - n_val
    
    train_idx = seq_time_idx[:n_train]
    val_idx = seq_time_idx[n_train:n_train + n_val]
    test_idx = seq_time_idx[n_train + n_val:]
    
    print(f"Temporal splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    
    # Assert splits are contiguous and disjoint
    assert set(train_idx).isdisjoint(set(val_idx)), "Overlap between train and validation indices."
    assert set(val_idx).isdisjoint(set(test_idx)), "Overlap between val and test indices."
    assert train_idx[-1] < val_idx[0], "Splits are not chronologically ordered."
    assert val_idx[-1] < test_idx[0], "Splits are not chronologically ordered."
    
    # 7. Dataset Construction (38 Features)
    feature_names = get_feature_names(seq_len)
    print(f"Constructing dataset with {len(feature_names)} features...")
    
    # Train subsampled to speed up training
    X_train, y_train, t_train = build_pixel_dataset(
        rainfall, track_df, train_idx, CONFIG, subsample=CONFIG["max_train_pixels_per_step"], seed=CONFIG["random_seed"]
    )
    # Validation and test kept full to reconstruct the grid mapping
    X_val, y_val, t_val = build_pixel_dataset(rainfall, track_df, val_idx, CONFIG)
    X_test, y_test, t_test = build_pixel_dataset(rainfall, track_df, test_idx, CONFIG)
    
    print(f"Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # 8. Leakage Audit
    run_leakage_audit(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, CONFIG)
    
    # 9. Scaling Features (Strictly fitted on Train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 10. Fit Models
    # Handle NaNs in Pres for fitting (Tweedie/GBM can handle or drop)
    # Fill any NaNs in pressure (some IMD rows have missing pressure at landfall) with forward fill
    track_df["storm_pres_hpa"] = track_df["storm_pres_hpa"].ffill().bfill()
    # Re-run features block lookup if needed, but we can also just fill NaNs in features
    for X_arr in (X_train_scaled, X_val_scaled, X_test_scaled, X_train, X_val, X_test):
        col_pres_idx = feature_names.index("storm_pres_hpa")
        nan_mask = np.isnan(X_arr[:, col_pres_idx])
        if nan_mask.any():
            # Fill with average pressure (e.g. 950 hPa)
            X_arr[nan_mask, col_pres_idx] = 950.0
            
    models_dict = get_fitted_models(X_train_scaled, y_train, CONFIG)
    
    # Baselines
    baselines = get_baselines(X_test, feature_names)
    
    # Predict on val and test
    val_preds = {}
    test_preds = {}
    
    for name, model in models_dict.items():
        if name == "Two-Stage Model":
            val_preds[name] = model.predict(X_val_scaled)
            test_preds[name] = model.predict(X_test_scaled)
        else:
            # Predict log1p and convert back to mm
            val_preds[name] = np.clip(np.expm1(model.predict(X_val_scaled)), 0, None)
            test_preds[name] = np.clip(np.expm1(model.predict(X_test_scaled)), 0, None)
            
    # Add baselines
    for name, pred in baselines.items():
        test_preds[name] = pred
        
    # 11. Ensembles
    # Greedy selection weights
    selected_list, greedy_weights, greedy_val_rmse = greedy_ensemble_select(val_preds, y_val)
    print(f"Greedy Ensemble weights: {greedy_weights}")
    
    # Construct greedy ensemble test predictions
    greedy_test = np.zeros(X_test.shape[0])
    for name, w in greedy_weights.items():
        greedy_test += w * test_preds[name]
    test_preds["Greedy Ensemble"] = greedy_test
    
    # BMA weights
    bma_weights = compute_bma_weights(val_preds, y_val)
    print(f"BMA weights: { {k: round(v, 4) for k, v in bma_weights.items()} }")
    bma_test = np.zeros(X_test.shape[0])
    for name, w in bma_weights.items():
        bma_test += w * test_preds[name]
    test_preds["BMA Ensemble"] = bma_test
    
    # 12. Evaluate all model results
    rows = []
    # Evaluate baselines first
    for name in baselines:
        m = full_metrics(y_test, test_preds[name])
        rows.append({"Model": name, "Category": "Baseline", **m})
        
    # Evaluate ML models
    for name in models_dict:
        m = full_metrics(y_test, test_preds[name])
        rows.append({"Model": name, "Category": "Machine Learning", **m})
        
    # Evaluate Ensembles
    for name in ["Greedy Ensemble", "BMA Ensemble"]:
        m = full_metrics(y_test, test_preds[name])
        rows.append({"Model": name, "Category": "Ensemble", **m})
        
    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("TEST SPLIT LEADERBOARD")
    print("=" * 60)
    print(results_df.round(4).to_string(index=False))
    
    # Save leaderboard
    results_df.to_csv(Path(CONFIG["metrics_dir"]) / "model_leaderboard.csv", index=False)
    
    # 13. Ablation Study
    print("Running feature group ablation study...")
    # Definitions of indices
    lag_cols = [n for n in feature_names if n.startswith("lag_") or n.startswith("has_lag_") or n in ("roll_mean_4", "roll_max_4", "roll_min_4", "roll_std_4", "ema_4", "d1_trend", "d2_accel")]
    track_cols = ["storm_lat", "storm_lon", "storm_wind_kt", "storm_pres_hpa", "storm_trans_speed", "storm_heading", "track_age", "dist_to_storm", "azimuth_to_storm", "along_track_dist", "cross_track_dist"]
    geom_cols = ["dist_to_storm", "azimuth_to_storm", "along_track_dist", "cross_track_dist"]
    phys_cols = ["grad_lat", "grad_lon", "grad_mag", "laplacian", "local_mean_3x3", "dist_to_wet_centroid", "azimuth_to_wet_centroid", "no_rain_prev_flag"]
    
    # Feature group index mapper
    def get_indices(cols):
        return [feature_names.index(c) for c in cols if c in feature_names]
        
    ab_groups = {
        "A. Rain history only": get_indices(lag_cols),
        "B. Storm track only": get_indices(track_cols),
        "C. Rain + storm track": get_indices(lag_cols + track_cols),
        "D. Rain + storm-relative geometry": get_indices(lag_cols + geom_cols),
        "E. Rain + physics": get_indices(lag_cols + phys_cols),
        "F. All features (Full Set)": list(range(len(feature_names)))
    }
    
    from sklearn.ensemble import RandomForestRegressor
    ab_rows = []
    
    # Run ablation experiments
    for name, cols_idx in ab_groups.items():
        if len(cols_idx) == 0:
            continue
        X_tr_ab = X_train[:, cols_idx]
        X_te_ab = X_test[:, cols_idx]
        
        # Train estimator
        est = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=CONFIG["random_seed"])
        est.fit(X_tr_ab, np.log1p(y_train))
        pred_ab = np.clip(np.expm1(est.predict(X_te_ab)), 0, None)
        
        m_ab = full_metrics(y_test, pred_ab)
        ab_rows.append({
            "Experiment": name,
            "n_features": len(cols_idx),
            "RMSE": m_ab["RMSE"],
            "MAE": m_ab["MAE"],
            "CSI@5.0": m_ab["CSI@5.0"]
        })
        
    ablation_df = pd.DataFrame(ab_rows)
    print("\n" + "=" * 60)
    print("ABLATION STUDY STATISTICS")
    print("=" * 60)
    print(ablation_df.round(4).to_string(index=False))
    ablation_df.to_csv(Path(CONFIG["metrics_dir"]) / "ablation_results.csv", index=False)
    
    # 15. Permutation Feature Importance (using Extra Trees on train set)
    print("Calculating permutation feature importance...")
    et_model = models_dict["Extra Trees"]
    # Permutation importance computed on a validation subset to save time
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(et_model, X_val_scaled, np.log1p(y_val), n_repeats=3, random_state=CONFIG["random_seed"], n_jobs=-1)
    importance_series = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
    
    # 16. Paired Block Bootstrap Significance
    print("Performing paired-block bootstrap significance...")
    sig_rows = []
    model_keys = list(models_dict.keys())
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            m_a, m_b = model_keys[i], model_keys[j]
            mean_diff, ci, p = paired_block_bootstrap(y_test, test_preds[m_a], test_preds[m_b], t_test, n_boot=200)
            sig_rows.append({
                "Model A": m_a,
                "Model B": m_b,
                "Mean RMSE diff": mean_diff,
                "CI low": ci[0],
                "CI high": ci[1],
                "p_value": p,
                "Significant": p < 0.05
            })
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(Path(CONFIG["metrics_dir"]) / "pairwise_significance.csv", index=False)
    
    # 17. Conformal Prediction calibration check
    print("Checking conformal prediction calibration...")
    # Quantile regression calibration
    # Create 9 quantile levels
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    observed_coverage = []
    from sklearn.ensemble import HistGradientBoostingRegressor
    for q in quantile_levels:
        m_q = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=50, max_depth=4, random_state=42)
        m_q.fit(X_train_scaled, np.log1p(y_train))
        pred_q = np.clip(np.expm1(m_q.predict(X_test_scaled)), 0, None)
        coverage = float(np.mean(y_test <= pred_q))
        observed_coverage.append(coverage)
        
    # 18. Generate Diagnostics (Figures & Tables)
    plot_fig01_fani_track(track_df, CONFIG)
    plot_fig02_precip_spatial(rainfall, CONFIG)
    plot_fig03_temporal_evolution(rainfall, CONFIG)
    plot_fig04_rainfall_distribution(rainfall, CONFIG)
    plot_fig05_extreme_rainfall_map(rainfall, CONFIG)
    plot_fig06_observed_vs_predicted(y_test, test_preds["Greedy Ensemble"], "Greedy Ensemble", CONFIG)
    plot_fig07_residual_distribution(y_test, test_preds["Greedy Ensemble"], "Greedy Ensemble", CONFIG)
    plot_fig08_model_comparison(results_df[results_df["Category"] != "Baseline"], CONFIG)
    plot_fig09_storm_relative_structure(X_test, y_test, feature_names, CONFIG)
    plot_fig10_rain_vs_distance(X_test, y_test, feature_names, CONFIG)
    plot_fig11_rain_vs_intensity(X_test, y_test, feature_names, CONFIG)
    plot_fig12_rain_vs_speed(X_test, y_test, feature_names, CONFIG)
    plot_fig13_spatial_error(rainfall, y_test, test_preds["Greedy Ensemble"], t_test, CONFIG)
    plot_fig14_temporal_error(y_test, test_preds["Greedy Ensemble"], t_test, CONFIG)
    plot_fig15_extreme_performance(y_test, test_preds["Greedy Ensemble"], CONFIG)
    plot_fig16_uncertainty_calibration(observed_coverage, quantile_levels, CONFIG)
    plot_fig17_feature_importance(importance_series, CONFIG)
    
    # Construct ablation df matching FIG 18 needs
    ab_df_fig18 = pd.DataFrame([
        {"Removed group": name.split(". ")[1], "Delta RMSE": r["RMSE"] - results_df[results_df["Model"] == "Random Forest"]["RMSE"].values[0]}
        for name, r in zip(ab_groups.keys(), ab_rows)
    ])
    plot_fig18_ablation(ab_df_fig18, CONFIG)
    plot_fig19_case_study(rainfall, y_test, test_preds["Greedy Ensemble"], t_test, CONFIG)
    
    # Save Tables
    generate_tables(dq_report, results_df[results_df["Category"] == "Baseline"], 
                    results_df[results_df["Category"] == "Machine Learning"], ablation_df, 
                    sig_df, importance_series, CONFIG)
    
    # 19. Reproducibility metadata export
    repro_log = {
        "run_time_seconds": time.time() - start_time,
        "timestamps": str(pd.Timestamp.now()),
        "python_version": os.sys.version,
        "config": CONFIG,
        "features": feature_names,
        "best_weights": greedy_weights,
    }
    with open(Path(CONFIG["reproducibility_dir"]) / "reproducibility_report.json", "w") as f:
        json.dump(repro_log, f, indent=4, default=str)
        
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUCESSFULLY COMPLETED")
    print("=" * 80)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return results_df
