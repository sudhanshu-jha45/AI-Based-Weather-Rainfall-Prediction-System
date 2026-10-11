import os
from pathlib import Path

# Base workspace path
BASE_DIR = Path("c:/AI_Projects/Fani_Rainfall_Prediction")

CONFIG = {
    "random_seed": 42,
    "seq_len": 4,
    "forecast_horizons": [1],  # 1-hour ahead prediction
    "rain_thresholds": [1.0, 5.0, 10.0],  # mm/h
    "test_fraction": 0.2,
    "validation_fraction": 0.2,
    "primary_metric": "RMSE",
    "secondary_metrics": ["MAE", "Bias", "R2", "NSE", "KGE", "r"],
    "max_train_pixels_per_step": 20000,
    
    # Raw Data Paths
    "source_a_path": str(BASE_DIR / "8e6813e7d3921a9cf893370f578a6f33.nc"),
    "source_b_path": str(BASE_DIR / "be289ee5120227eb6248ab74b83d33e.nc"),
    "source_c_path": str(BASE_DIR / "data_stream-oper_stepType-accum.nc"),
    "track_path": str(BASE_DIR / "fani_official_besttrack_IMD.csv"),
    
    # Output Directories
    "output_dir": "outputs",
    "data_quality_dir": "outputs/data_quality",
    "leakage_reports_dir": "outputs/leakage_reports",
    "processed_data_dir": "outputs/processed_data",
    "predictions_dir": "outputs/predictions",
    "metrics_dir": "outputs/metrics",
    "model_dir": "outputs/models",
    "figure_dir": "outputs/figures",
    "legacy_figure_dir": "outputs/figures/legacy",
    "corrected_figure_dir": "outputs/figures/corrected",
    "paper_figure_dir": "outputs/figures/paper",
    "table_dir": "outputs/tables",
    "experiment_metadata_dir": "outputs/experiment_metadata",
    "reproducibility_dir": "outputs/reproducibility",
}

# Ensure all output directories exist
for key, path in CONFIG.items():
    if key.endswith("_dir") and isinstance(path, str):
        os.makedirs(Path(BASE_DIR / path), exist_ok=True)

# Station Reference Locations
KEY_LOCATIONS = {
    "Puri":        (19.80, 85.83),
    "Bhubaneswar": (20.27, 85.84),
    "Cuttack":     (20.47, 85.88),
    "Balasore":    (21.49, 86.93),
    "Kendrapara":  (20.50, 86.42),
}

# Reference Landfall
LANDFALL_TIME = "2019-05-03 03:30:00"
LANDFALL_LOC = {"lat": 19.75, "lon": 85.70, "name": "Landfall (near Puri)"}
