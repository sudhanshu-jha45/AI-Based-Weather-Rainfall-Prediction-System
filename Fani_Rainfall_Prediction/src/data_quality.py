import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

def run_data_quality_checks(rainfall, reset_mask, config):
    """Perform dataset checks and output reports.
    
    Verifies:
      - Coordinate monotonicity and bounds.
      - Absence of duplicates or missing timestamps.
      - Absence of NaNs.
      - Negative value counts and extremes.
      - Monotonicity of spatial axes.
    """
    print("=" * 80)
    print("DATA QUALITY AUDIT")
    print("=" * 80)

    timestamps = pd.DatetimeIndex(rainfall.time.values)
    lats = rainfall.latitude.values
    lons = rainfall.longitude.values
    rain_vals = rainfall.values

    # 1. Assertions
    assert timestamps.is_monotonic_increasing, "Timestamps are not monotonically increasing."
    assert not timestamps.duplicated().any(), "Duplicate timestamps found."
    assert np.isfinite(lats).all(), "Non-finite values found in latitude."
    assert np.isfinite(lons).all(), "Non-finite values found in longitude."
    assert ((lats >= -90) & (lats <= 90)).all(), "Latitude values outside bounds [-90, 90]."
    assert ((lons >= -360) & (lons <= 360)).all(), "Longitude values outside bounds [-360, 360]."

    # 2. Checks
    total_steps = len(timestamps)
    has_nans = bool(np.isnan(rain_vals).any())
    nan_count = int(np.isnan(rain_vals).sum())
    neg_count = int(np.sum(rain_vals < 0))
    zero_fraction = float(np.sum(rain_vals == 0) / rain_vals.size)

    min_val = float(np.nanmin(rain_vals))
    max_val = float(np.nanmax(rain_vals))
    mean_val = float(np.nanmean(rain_vals))
    std_val = float(np.nanstd(rain_vals))
    
    time_diffs = np.diff(timestamps).astype('timedelta64[h]').astype(float)
    irregular_intervals = int(np.sum(time_diffs != 1.0))
    
    # 3. Compile report dict
    report = {
        "dataset_name": "Cyclone Fani ERA5 Rainfall",
        "time_start": str(timestamps[0]),
        "time_end": str(timestamps[-1]),
        "total_timesteps": total_steps,
        "grid_shape": [len(lats), len(lons)],
        "lat_range": [float(lats.min()), float(lats.max())],
        "lon_range": [float(lons.min()), float(lons.max())],
        "has_nans": has_nans,
        "nan_count": nan_count,
        "negative_rainfall_count": neg_count,
        "zero_fraction": zero_fraction,
        "min_rainfall_mm": min_val,
        "max_rainfall_mm": max_val,
        "mean_rainfall_mm": mean_val,
        "std_rainfall_mm": std_val,
        "accumulation_resets_detected": int(np.sum(reset_mask)),
        "irregular_time_intervals": irregular_intervals,
        "lat_is_monotonic": bool(pd.Index(lats).is_monotonic_decreasing or pd.Index(lats).is_monotonic_increasing),
        "lon_is_monotonic": bool(pd.Index(lons).is_monotonic_decreasing or pd.Index(lons).is_monotonic_increasing),
    }

    # Print human-readable report
    for k, v in report.items():
        print(f"  {k:30s}: {v}")
    print("=" * 80)

    # Save to JSON
    dq_dir = Path(config["data_quality_dir"])
    os.makedirs(dq_dir, exist_ok=True)
    
    json_path = dq_dir / "fani_data_quality_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
        
    # Save to CSV
    csv_path = dq_dir / "fani_data_quality_report.csv"
    pd.DataFrame([report]).to_csv(csv_path, index=False)
    
    print(f"Saved Data Quality Report to:\n  - {json_path}\n  - {csv_path}")
    return report
