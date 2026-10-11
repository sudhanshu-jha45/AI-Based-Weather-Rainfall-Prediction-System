import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def load_and_clean_track(track_csv_path):
    """Load IMD track CSV, standardize columns, and causally compute dynamics.
    
    Standardized columns:
      - time (pd.Timestamp)
      - storm_lat (float)
      - storm_lon (float)
      - storm_wind_kt (float)
      - storm_pres_hpa (float)
      - storm_trans_speed (float, km/h)
      - storm_heading (float, rad)
    """
    df = pd.read_csv(track_csv_path)
    
    # 1. Standardize columns
    df = df.rename(columns={
        "datetime_utc": "time",
        "lat": "storm_lat",
        "lon": "storm_lon",
        "max_sustained_wind_kt": "storm_wind_kt",
        "central_pressure_hpa": "storm_pres_hpa"
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    # 2. Causal dynamics precomputation (speed, heading)
    trans_speed = [0.0]
    heading = [0.0]
    
    for i in range(1, len(df)):
        dt = (df.loc[i, "time"] - df.loc[i-1, "time"]).total_seconds() / 3600.0
        if dt <= 0:
            trans_speed.append(0.0)
            heading.append(0.0)
            continue
            
        dlat = df.loc[i, "storm_lat"] - df.loc[i-1, "storm_lat"]
        dlon = df.loc[i, "storm_lon"] - df.loc[i-1, "storm_lon"]
        mid_lat = (df.loc[i, "storm_lat"] + df.loc[i-1, "storm_lat"]) / 2.0
        
        dy = dlat * 111.0  # km
        dx = dlon * 111.3 * np.cos(np.radians(mid_lat))  # km
        
        speed = np.sqrt(dx**2 + dy**2) / dt  # km/h
        angle = np.arctan2(dx, dy)  # heading angle in radians
        
        trans_speed.append(float(speed))
        heading.append(float(angle))
        
    df["storm_trans_speed"] = trans_speed
    df["storm_heading"] = heading
    
    # Do not fabricate missing pressure (there might be NaNs like in landfall row)
    # The grade column is preserved as storm_grade
    if "grade" in df.columns:
        df["storm_grade"] = df["grade"]
        
    return df[["time", "storm_lat", "storm_lon", "storm_wind_kt", "storm_pres_hpa", "storm_trans_speed", "storm_heading"]]

def build_spatial_derivatives(frame):
    """Compute spatial derivatives on (H, W) frame using padded finite differences."""
    padded = np.pad(frame, 1, mode="edge")
    
    # First derivatives (centered difference)
    grad_lat = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
    grad_lon = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    
    # Second derivative (Laplacian)
    laplacian = (padded[2:, 1:-1] + padded[:-2, 1:-1] +
                 padded[1:-1, 2:] + padded[1:-1, :-2] - 4.0 * frame)
    
    grad_mag = np.sqrt(grad_lat ** 2 + grad_lon ** 2)
    
    # Local 3x3 mean
    local_mean = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0
    
    return grad_lat, grad_lon, grad_mag, laplacian, local_mean

def build_centroid_features(frame, lat2d, lon2d):
    """Compute distance and bearing to the wet centroid of the frame."""
    wsum = float(frame.sum())
    if wsum > 1e-6:
        lat_c = float((frame * lat2d).sum() / wsum)
        lon_c = float((frame * lon2d).sum() / wsum)
        no_rain_flag = 0.0
    else:
        lat_c = float(np.mean(lat2d))
        lon_c = float(np.mean(lon2d))
        no_rain_flag = 1.0
        
    coslat = np.cos(np.radians(lat2d))
    dlat = lat2d - lat_c
    dlon_eff = (lon2d - lon_c) * coslat
    
    dist = np.sqrt(dlat ** 2 + dlon_eff ** 2) * 111.0  # approximate km
    azimuth = (np.degrees(np.arctan2(dlon_eff, dlat)) + 360.0) % 360.0
    
    return dist, azimuth, no_rain_flag

def build_features_at_t(field, lat2d, lon2d, t, seq_len, aligned_track_row, deep_lags=(6, 12)):
    """Assemble all physical local, spatial, and storm-relative features at step t.
    
    Uses only history field[:t] to enforce causality.
    """
    H, W = field.shape[1], field.shape[2]
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)
    
    # 1. Temporal lags (SEQ_LEN = 4)
    # Lags are in order: lag_4, lag_3, lag_2, lag_1
    lags_seq = [field[t - (seq_len - i)] for i in range(seq_len)]
    lag_stack = np.stack([l.reshape(-1) for l in lags_seq], axis=1)
    
    # Cumulative stats
    cum_before = field[:t].sum(axis=0).reshape(-1)
    cum_rate = cum_before / max(t, 1)
    
    # Rolling stats (mean, max, min, std)
    roll_mean = lag_stack.mean(axis=1)
    roll_max = lag_stack.max(axis=1)
    roll_min = lag_stack.min(axis=1)
    roll_std = lag_stack.std(axis=1)
    
    # Exponential moving average (EMA)
    w = np.array([0.0625, 0.125, 0.25, 0.5])[:seq_len]
    ema_w = w / w.sum()
    ema = (lag_stack * ema_w).sum(axis=1)
    
    # Trends
    lag1 = lag_stack[:, -1]
    lag2 = lag_stack[:, -2] if seq_len >= 2 else lag1
    lag3 = lag_stack[:, -3] if seq_len >= 3 else lag2
    d1 = lag1 - lag2
    d2 = (lag1 - lag2) - (lag2 - lag3)
    
    # Deep lags (e.g. lag_6, lag_12)
    deep_feats = []
    for k in deep_lags:
        if t - k >= 0:
            lk = field[t - k].reshape(-1)
            flag = np.ones(lk.shape[0])
        else:
            lk = lag_stack[:, 0]  # fallback to oldest available
            flag = np.zeros(lag_stack.shape[0])
        deep_feats.extend([lk, flag])
        
    # 2. Spatial derivatives and wet centroid on t-1 frame
    frame_prev = field[t - 1]
    grad_lat, grad_lon, grad_mag, laplacian, local_mean = build_spatial_derivatives(frame_prev)
    dist_c, azimuth_c, no_rain_flag = build_centroid_features(frame_prev, lat2d, lon2d)
    
    spatial_feats = [
        grad_lat.reshape(-1), grad_lon.reshape(-1), grad_mag.reshape(-1),
        laplacian.reshape(-1), local_mean.reshape(-1),
        dist_c.reshape(-1), azimuth_c.reshape(-1),
        np.full(lat_flat.shape[0], no_rain_flag)
    ]
    
    # 3. Storm-relative physics-informed features
    # Retrieve causally-aligned storm track information
    s_lat = aligned_track_row["storm_lat"]
    s_lon = aligned_track_row["storm_lon"]
    s_wind = aligned_track_row["storm_wind_kt"]
    s_pres = aligned_track_row["storm_pres_hpa"]
    s_speed = aligned_track_row["storm_trans_speed"]
    s_heading = aligned_track_row["storm_heading"]
    track_age = aligned_track_row["track_age"]
    
    # Spatial displacements to storm center
    dlat = lat_flat - s_lat
    dlon = lon_flat - s_lon
    dy = dlat * 111.0  # km
    dx = dlon * 111.3 * np.cos(np.radians((lat_flat + s_lat) / 2.0))  # km
    
    dist_to_storm = np.sqrt(dx**2 + dy**2)
    azimuth_to_storm = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    
    # Along-track and cross-track projection (rotated coordinates)
    along_track = dx * np.sin(s_heading) + dy * np.cos(s_heading)
    cross_track = -dx * np.cos(s_heading) + dy * np.sin(s_heading)
    
    storm_feats = [
        np.full(lat_flat.shape[0], s_lat),
        np.full(lat_flat.shape[0], s_lon),
        np.full(lat_flat.shape[0], s_wind),
        np.full(lat_flat.shape[0], s_pres),
        np.full(lat_flat.shape[0], s_speed),
        np.full(lat_flat.shape[0], s_heading),
        np.full(lat_flat.shape[0], track_age),
        dist_to_storm,
        azimuth_to_storm,
        along_track,
        cross_track
    ]
    
    # Combine everything
    feat_cols = (
        [lag_stack[:, i] for i in range(seq_len)] +
        [lat_flat, lon_flat, cum_before, cum_rate,
         roll_mean, roll_max, roll_min, roll_std, ema, d1, d2] +
        deep_feats + spatial_feats + storm_feats
    )
    return np.column_stack(feat_cols)

def get_feature_names(seq_len, deep_lags=(6, 12)):
    """Generate standardized feature names corresponding to feature array columns."""
    names = [f"lag_{seq_len - i}" for i in range(seq_len)] + [
        "lat", "lon", "cum_rain_before", "cum_rain_rate",
        "roll_mean_4", "roll_max_4", "roll_min_4", "roll_std_4", "ema_4",
        "d1_trend", "d2_accel"
    ]
    for k in deep_lags:
        names.extend([f"lag_{k}", f"has_lag_{k}"])
    names.extend([
        "grad_lat", "grad_lon", "grad_mag", "laplacian", "local_mean_3x3",
        "dist_to_wet_centroid", "azimuth_to_wet_centroid", "no_rain_prev_flag"
    ])
    names.extend([
        "storm_lat", "storm_lon", "storm_wind_kt", "storm_pres_hpa",
        "storm_trans_speed", "storm_heading", "track_age",
        "dist_to_storm", "azimuth_to_storm", "along_track_dist", "cross_track_dist"
    ])
    return names

def build_pixel_dataset(rainfall, track_df, time_indices, config, subsample=None, seed=42):
    """Construct feature matrix X and target vector y for the specified time indices.
    
    Maintains a strict causal structure. Track information is causally aligned.
    """
    rng = np.random.default_rng(seed)
    seq_len = config["seq_len"]
    
    lat2d, lon2d = np.meshgrid(rainfall.latitude.values, rainfall.longitude.values, indexing="ij")
    field = rainfall.values.astype("float32")
    
    # 1. Causal track alignment via merge_asof
    rain_times_df = pd.DataFrame({"time": pd.to_datetime(rainfall.time.values)})
    
    # Perform backward merge to align track causally
    aligned_track = pd.merge_asof(
        rain_times_df,
        track_df,
        on="time",
        direction="backward"
    )
    
    # Calculate track age in hours
    aligned_track["track_age"] = (aligned_track["time"] - aligned_track["time"]).dt.total_seconds() / 3600.0
    for i in range(len(aligned_track)):
        # Match time from merged row back to find age
        matched_time = aligned_track.loc[i, "time"]
        # Find latest track observation <= matched_time
        track_obs = track_df[track_df["time"] <= matched_time]
        if len(track_obs) > 0:
            latest_t = track_obs.iloc[-1]["time"]
            aligned_track.loc[i, "track_age"] = (matched_time - latest_t).total_seconds() / 3600.0
            
    X_parts, y_parts, t_parts = [], [], []
    
    for t in time_indices:
        # Fetch causally aligned track row at index t
        track_row = aligned_track.iloc[t].to_dict()
        
        # Build features for all grid pixels at step t
        feat = build_features_at_t(field, lat2d, lon2d, t, seq_len, track_row)
        target = field[t].reshape(-1)
        
        n_pix = feat.shape[0]
        idx = np.arange(n_pix)
        if subsample is not None and n_pix > subsample:
            idx = rng.choice(n_pix, size=subsample, replace=False)
            
        X_parts.append(feat[idx])
        y_parts.append(target[idx])
        t_parts.append(np.full(len(idx), t))
        
    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(t_parts)
