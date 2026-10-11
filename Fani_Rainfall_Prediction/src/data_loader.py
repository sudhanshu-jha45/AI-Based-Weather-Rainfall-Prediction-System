import xarray as xr
import pandas as pd
import numpy as np
import os

def _open_fani_nc(path):
    """Open a Fani NetCDF robustly, trying multiple engines."""
    errors = []
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as e:
            errors.append(f"{engine}: {type(e).__name__}: {e}")
    raise OSError(
        f"Could not open {path} with available NetCDF engines.\n" +
        "\n".join(errors)
    )

def _normalise_fani_raw(ds_in, source_name):
    """Normalize coordinate names, handle expver and number dimensions, and sort time."""
    ds0 = ds_in.copy()

    # Squeeze number dimension
    if "number" in ds0.dims:
        if ds0.sizes["number"] == 1:
            ds0 = ds0.squeeze("number", drop=True)
        else:
            ds0 = ds0.isel(number=0)
    elif "number" in ds0.coords:
        ds0 = ds0.reset_coords("number", drop=True)

    # Handle ERA5 expver dimension (contains ERA5 and ERA5T)
    if "expver" in ds0.dims:
        if ds0.sizes["expver"] > 1:
            vals = list(ds0["expver"].values)
            if 1 in vals:
                primary = ds0.sel(expver=1)
                fallback_id = 5 if 5 in vals else vals[-1]
                ds0 = primary.combine_first(ds0.sel(expver=fallback_id))
            else:
                ds0 = ds0.isel(expver=0)
        else:
            ds0 = ds0.squeeze("expver", drop=True)
    elif "expver" in ds0.coords:
        ds0 = ds0.reset_coords("expver", drop=True)

    # Standardize valid_time to time
    if "valid_time" in ds0.dims:
        ds0 = ds0.rename({"valid_time": "time"})
    elif "valid_time" in ds0.coords and "time" not in ds0.dims:
        ds0 = ds0.rename({"valid_time": "time"})

    if "time" not in ds0.dims:
        raise KeyError(
            f"{source_name}: no time/valid_time dimension found. Dimensions: {dict(ds0.dims)}"
        )

    if "latitude" not in ds0.dims or "longitude" not in ds0.dims:
        raise KeyError(
            f"{source_name}: expected latitude/longitude dimensions. Dimensions: {dict(ds0.dims)}"
        )

    if "tp" not in ds0.data_vars:
        raise KeyError(
            f"{source_name}: variable 'tp' not found. Variables: {list(ds0.data_vars)}"
        )

    ds0 = ds0.sortby("time")

    # Remove duplicate timestamps
    time_index = pd.DatetimeIndex(ds0.time.values)
    duplicate_mask = time_index.duplicated(keep="first")
    if duplicate_mask.any():
        ds0 = ds0.isel(time=~duplicate_mask)
        print(f"{source_name}: removed {int(duplicate_mask.sum())} duplicate timestamp(s).")

    return ds0[["tp"]]

def load_and_merge_datasets(source_a_path, source_b_path):
    """Load Source A and B, check consistency, and merge along time."""
    print("Loading netCDF files...")
    raw_b = _open_fani_nc(source_b_path)
    raw_a = _open_fani_nc(source_a_path)

    try:
        Bn = _normalise_fani_raw(raw_b, "Source B")
        An = _normalise_fani_raw(raw_a, "Source A")

        # Grid consistency check
        if not np.array_equal(Bn.latitude.values, An.latitude.values):
            raise ValueError("Source A and Source B latitude grids differ.")

        if not np.array_equal(Bn.longitude.values, An.longitude.values):
            raise ValueError("Source A and Source B longitude grids differ.")

        # Temporal transition check
        b_end = pd.Timestamp(Bn.time.values[-1])
        a_start = pd.Timestamp(An.time.values[0])
        gap_hours = (a_start - b_end).total_seconds() / 3600.0
        print(f"Source B coverage: {pd.Timestamp(Bn.time.values[0])} to {b_end}")
        print(f"Source A coverage: {a_start} to {pd.Timestamp(An.time.values[-1])}")
        print(f"Gap between Source B end and Source A start: {gap_hours:.2f} hour(s)")

        # Merge datasets along time
        merged = xr.concat(
            [Bn, An],
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override"
        ).sortby("time")

        # Remove duplicate timestamps from merged record
        duplicate_count = int(merged.time.to_index().duplicated().sum())
        if duplicate_count > 0:
            merged = merged.isel(
                time=~merged.time.to_index().duplicated(keep="first")
            )
            print(f"Merged dataset: removed {duplicate_count} duplicate timestamp(s).")

        return merged

    finally:
        raw_b.close()
        raw_a.close()

def deaccumulate_rainfall(raw_ds):
    """Deaccumulates total precipitation (meters) to step rainfall (mm).
    
    tp_mm(t) = tp(t)*1000.
    Calculates differences: diffs(t) = tp_mm(t) - tp_mm(t-1).
    Resets occur when tp_mm decreases. In those cases, the hourly rainfall is the
    accumulated amount itself, i.e. tp_mm(t).
    """
    tp_m = raw_ds["tp"]
    tp_mm = (tp_m * 1000.0).astype("float32")
    tp_mm.attrs["units"] = "mm"
    tp_mm.name = "tp_mm"

    vals = tp_mm.values.copy()
    diffs = np.diff(vals, axis=0)

    # Scans for accumulation resets (decreases in accumulated values)
    reset_mask = np.zeros(vals.shape[0], dtype=bool)
    if vals.shape[0] > 1:
        reset_mask[1:] = np.any(diffs < -1e-8, axis=(1, 2))

    step_vals = np.empty_like(vals)
    step_vals[0] = vals[0]  # first timestep: assume accumulated value as rainfall

    if vals.shape[0] > 1:
        step_vals[1:] = np.where(
            reset_mask[1:, None, None],
            vals[1:],
            diffs
        )

    # Numerical safety: precipitation cannot be negative
    neg_count = int(np.sum(step_vals < 0))
    if neg_count > 0:
        print(f"Data Quality warning: clipped {neg_count} negative deaccumulated values to 0.")
    step_vals = np.clip(step_vals, 0, None)

    rainfall = tp_mm.copy(data=step_vals)
    rainfall.name = "rainfall_mm"
    rainfall.attrs.update(
        units="mm",
        long_name="Hourly rainfall per time step (de-accumulated from ERA5 total precipitation)"
    )
    
    return rainfall, reset_mask
