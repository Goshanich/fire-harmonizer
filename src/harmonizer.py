"""4-stage MODIS and VIIRS active fire harmonization engine.

Implements NASA FIRMS schema unification & confidence filtering, pixel swath-area
and FRP normalization, 0.01 deg (~1 km) spatiotemporal gridding, and overlap
cross-calibration (2012–2025) with bootstrap 95% confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from src.config import (
    DEFAULT_MODIS_MIN_CONFIDENCE,
    DEFAULT_VIIRS_CONFIDENCE,
    GRID_STEP_DEG,
    OVERLAP_YEARS,
    VIIRS_CONFIDENCE_WEIGHTS,
    VIIRS_TRANSITION_DATE,
)

_UNIFIED_COLS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "daynight",
    "scan",
    "track",
    "frp",
    "brightness_k",
    "confidence_raw",
    "confidence_weight",
    "sensor_group",
    "satellite",
    "instrument",
]

_VIIRS_CONF_ALIASES = {
    "low": "l",
    "nominal": "n",
    "high": "h",
    "l": "l",
    "n": "n",
    "h": "h",
}


@dataclass
class HarmonizationResult:
    """Structured output of the 4-stage MODIS/VIIRS harmonization pipeline."""

    filtered_points_df: pd.DataFrame
    gridded_df: pd.DataFrame
    calibration_df: pd.DataFrame
    monthly_df: pd.DataFrame
    harmonized_cells_df: pd.DataFrame
    overlap_monthly_df: pd.DataFrame
    transition_metrics: dict[str, float]


def _normalize_viirs_confidence(series: pd.Series) -> pd.Series:
    """Normalize VIIRS categorical confidence labels to 'l', 'n', or 'h'."""
    str_vals = series.fillna("n").astype(str).str.strip().str.lower()
    mapped = str_vals.map(_VIIRS_CONF_ALIASES)
    if mapped.isna().any():
        numeric_vals = pd.to_numeric(str_vals, errors="coerce")
        is_num = mapped.isna() & numeric_vals.notna()
        mapped = mapped.where(
            ~is_num,
            np.where(numeric_vals >= 80, "h", np.where(numeric_vals >= 30, "n", "l")),
        )
    return mapped.fillna("l")


def unify_and_filter_sensors(
    modis_df: pd.DataFrame,
    viirs_df: pd.DataFrame,
    modis_min_conf: int = DEFAULT_MODIS_MIN_CONFIDENCE,
    viirs_allowed_conf: Optional[Iterable[str]] = None,
    daynight_filter: str = "All",
) -> pd.DataFrame:
    """Step 1: Unify MODIS and VIIRS schemas and filter low-confidence detections."""
    allowed_viirs = [
        str(c).strip().lower()
        for c in (viirs_allowed_conf if viirs_allowed_conf is not None else DEFAULT_VIIRS_CONFIDENCE)
    ]
    frames: list[pd.DataFrame] = []

    if modis_df is not None and not modis_df.empty:
        m = modis_df.copy()
        m.columns = [str(c).strip().lower() for c in m.columns]
        if daynight_filter in ("D", "N") and "daynight" in m.columns:
            m = m[m["daynight"] == daynight_filter]

        if not m.empty:
            conf_num = pd.to_numeric(m.get("confidence", 50), errors="coerce").fillna(0.0)
            m = m[conf_num >= float(modis_min_conf)].copy()
            if not m.empty:
                conf_kept = pd.to_numeric(m.get("confidence", 50), errors="coerce").fillna(50.0)
                bright_col = "brightness" if "brightness" in m.columns else "bright_ti4"
                m_out = pd.DataFrame(
                    {
                        "latitude": pd.to_numeric(m["latitude"], errors="coerce"),
                        "longitude": pd.to_numeric(m["longitude"], errors="coerce"),
                        "acq_date": m["acq_date"].astype(str),
                        "acq_time": m.get("acq_time", "1200").astype(str).str.zfill(4),
                        "daynight": m.get("daynight", "D").astype(str).str.upper(),
                        "scan": pd.to_numeric(m.get("scan", 1.0), errors="coerce").fillna(1.0),
                        "track": pd.to_numeric(m.get("track", 1.0), errors="coerce").fillna(1.0),
                        "frp": pd.to_numeric(m["frp"], errors="coerce").fillna(0.0).clip(lower=0.0),
                        "brightness_k": pd.to_numeric(m.get(bright_col, np.nan), errors="coerce"),
                        "confidence_raw": conf_kept.astype(int).astype(str),
                        "confidence_weight": np.clip(conf_kept / 100.0, 0.0, 1.0),
                        "sensor_group": "MODIS",
                        "satellite": m.get("satellite", "Terra/Aqua").astype(str),
                        "instrument": "MODIS",
                    }
                )
                frames.append(m_out.dropna(subset=["latitude", "longitude", "acq_date"]))

    if viirs_df is not None and not viirs_df.empty:
        v = viirs_df.copy()
        v.columns = [str(c).strip().lower() for c in v.columns]
        if daynight_filter in ("D", "N") and "daynight" in v.columns:
            v = v[v["daynight"] == daynight_filter]

        if not v.empty:
            conf_norm = _normalize_viirs_confidence(v.get("confidence", pd.Series("n", index=v.index)))
            v = v[conf_norm.isin(allowed_viirs)].copy()
            if not v.empty:
                conf_kept = _normalize_viirs_confidence(
                    v.get("confidence", pd.Series("n", index=v.index))
                )
                bright_col = "bright_ti4" if "bright_ti4" in v.columns else "brightness"
                v_out = pd.DataFrame(
                    {
                        "latitude": pd.to_numeric(v["latitude"], errors="coerce"),
                        "longitude": pd.to_numeric(v["longitude"], errors="coerce"),
                        "acq_date": v["acq_date"].astype(str),
                        "acq_time": v.get("acq_time", "1200").astype(str).str.zfill(4),
                        "daynight": v.get("daynight", "D").astype(str).str.upper(),
                        "scan": pd.to_numeric(v.get("scan", 0.45), errors="coerce").fillna(0.45),
                        "track": pd.to_numeric(v.get("track", 0.45), errors="coerce").fillna(0.45),
                        "frp": pd.to_numeric(v["frp"], errors="coerce").fillna(0.0).clip(lower=0.0),
                        "brightness_k": pd.to_numeric(v.get(bright_col, np.nan), errors="coerce"),
                        "confidence_raw": conf_kept,
                        "confidence_weight": conf_kept.map(VIIRS_CONFIDENCE_WEIGHTS).fillna(0.75),
                        "sensor_group": "VIIRS",
                        "satellite": v.get("satellite", "SNPP").astype(str),
                        "instrument": "VIIRS",
                    }
                )
                frames.append(v_out.dropna(subset=["latitude", "longitude", "acq_date"]))

    if not frames:
        return pd.DataFrame(columns=_UNIFIED_COLS)
    return pd.concat(frames, ignore_index=True)


def normalize_pixel_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Calculate effective swath pixel area (km^2) and FRP surface density (MW/km^2)."""
    out = df.copy()
    if out.empty:
        out["pixel_area_km2"] = pd.Series(dtype=float)
        out["frp_density"] = pd.Series(dtype=float)
        out["weighted_frp"] = pd.Series(dtype=float)
        return out

    scan = pd.to_numeric(out["scan"], errors="coerce").fillna(1.0).clip(lower=0.1)
    track = pd.to_numeric(out["track"], errors="coerce").fillna(1.0).clip(lower=0.1)
    out["pixel_area_km2"] = np.maximum(scan * track, 0.05)
    out["frp_density"] = out["frp"] / out["pixel_area_km2"]
    out["weighted_frp"] = out["frp"] * out["confidence_weight"]
    return out


def grid_spatiotemporal_clusters(
    df: pd.DataFrame,
    grid_step: float = GRID_STEP_DEG,
) -> pd.DataFrame:
    """Step 3: Aggregate sub-pixel VIIRS clusters and MODIS detections onto a 0.01 deg grid."""
    cols = [
        "sensor_group",
        "grid_lat",
        "grid_lon",
        "cell_center_lat",
        "cell_center_lon",
        "acq_date",
        "year",
        "month",
        "daynight",
        "active_cell_count",
        "raw_point_count",
        "frp_sum",
        "frp_mean",
        "frp_density_mean",
        "weighted_frp_sum",
        "pixel_area_km2_sum",
        "brightness_max_k",
        "confidence_weight_mean",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    work = df.copy()
    work["grid_lat"] = np.round(np.floor(work["latitude"] / grid_step) * grid_step, 4)
    work["grid_lon"] = np.round(np.floor(work["longitude"] / grid_step) * grid_step, 4)
    work["cell_center_lat"] = np.round(work["grid_lat"] + 0.5 * grid_step, 5)
    work["cell_center_lon"] = np.round(work["grid_lon"] + 0.5 * grid_step, 5)

    dt = pd.to_datetime(work["acq_date"], errors="coerce")
    work = work[dt.notna()].copy()
    dt = dt[dt.notna()]
    work["year"] = dt.dt.year.astype(int)
    work["month"] = dt.dt.month.astype(int)

    group_keys = [
        "sensor_group",
        "grid_lat",
        "grid_lon",
        "cell_center_lat",
        "cell_center_lon",
        "acq_date",
        "year",
        "month",
        "daynight",
    ]
    gridded = (
        work.groupby(group_keys, as_index=False)
        .agg(
            raw_point_count=("frp", "count"),
            frp_sum=("frp", "sum"),
            frp_mean=("frp", "mean"),
            frp_density_mean=("frp_density", "mean"),
            weighted_frp_sum=("weighted_frp", "sum"),
            pixel_area_km2_sum=("pixel_area_km2", "sum"),
            brightness_max_k=("brightness_k", "max"),
            confidence_weight_mean=("confidence_weight", "mean"),
        )
    )
    gridded["active_cell_count"] = 1
    return gridded[cols].sort_values(["acq_date", "sensor_group"]).reset_index(drop=True)


def _bootstrap_median_ratio_ci(
    modis_vals: np.ndarray,
    viirs_vals: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for Median(MODIS) / Median(VIIRS)."""
    n = len(modis_vals)
    if n < 2 or np.median(viirs_vals) <= 0:
        ratio = float(np.median(modis_vals) / max(np.median(viirs_vals), 1e-6)) if n > 0 else 1.0
        return ratio, ratio

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    m_med = np.median(modis_vals[idx], axis=1)
    v_med = np.maximum(np.median(viirs_vals[idx], axis=1), 1e-6)
    ratios = m_med / v_med
    low, high = np.percentile(ratios, [2.5, 97.5])
    return float(low), float(high)


def compute_overlap_calibration(
    gridded_df: pd.DataFrame,
    overlap_years: tuple[int, int] = OVERLAP_YEARS,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Step 4a: Derive monthly & day/night alpha(m, daynight) factors on overlap (2012–2025)."""
    y_start, y_end = overlap_years
    overlap_raw = gridded_df[
        (gridded_df["acq_date"] >= VIIRS_TRANSITION_DATE)
        & (gridded_df["year"] >= y_start)
        & (gridded_df["year"] <= y_end)
    ]

    def _agg_sensor(sensor_name: str) -> pd.DataFrame:
        sub = overlap_raw[overlap_raw["sensor_group"] == sensor_name]
        prefix = sensor_name.lower()
        if sub.empty:
            return pd.DataFrame(
                {
                    "year": pd.Series(dtype="int64"),
                    "month": pd.Series(dtype="int64"),
                    "daynight": pd.Series(dtype="object"),
                    f"{prefix}_active_cells": pd.Series(dtype="float64"),
                    f"{prefix}_raw_points": pd.Series(dtype="float64"),
                    f"{prefix}_frp_sum": pd.Series(dtype="float64"),
                }
            )
        return sub.groupby(["year", "month", "daynight"], as_index=False).agg(
            **{
                f"{prefix}_active_cells": ("active_cell_count", "sum"),
                f"{prefix}_raw_points": ("raw_point_count", "sum"),
                f"{prefix}_frp_sum": ("frp_sum", "sum"),
            }
        )

    m_ov = _agg_sensor("MODIS")
    v_ov = _agg_sensor("VIIRS")

    paired = pd.merge(m_ov, v_ov, on=["year", "month", "daynight"], how="outer").fillna(0.0)
    paired_valid = paired[paired["viirs_active_cells"] > 0].copy()

    default_alpha_by_dn: dict[str, float] = {"D": 0.85, "N": 0.36}
    default_alpha_overall: float = 0.60

    def _safe_median_ratios(df_sub: pd.DataFrame) -> tuple[float, float, float]:
        med_v_cells = float(np.median(df_sub["viirs_active_cells"]))
        med_m_cells = float(np.median(df_sub["modis_active_cells"]))
        a_cells = float(med_m_cells / max(med_v_cells, 1e-6))

        med_m_frp = float(np.median(df_sub["modis_frp_sum"]))
        med_v_frp = float(np.median(df_sub["viirs_frp_sum"]))
        a_frp = float(med_m_frp / max(med_v_frp, 1e-6)) if med_m_frp > 0 and med_v_frp > 0 else a_cells

        med_m_raw = float(np.median(df_sub["modis_raw_points"]))
        med_v_raw = float(np.median(df_sub["viirs_raw_points"]))
        r_raw = float(med_m_raw / max(med_v_raw, 1e-6)) if med_m_raw > 0 and med_v_raw > 0 else a_cells
        return a_cells, a_frp, r_raw

    # Global and day/night fallback ratios in case a specific (month, daynight) slice is sparse
    coactive_all = paired_valid[paired_valid["modis_active_cells"] > 0]
    has_global_modis = True
    if (
        not paired_valid.empty
        and np.median(paired_valid["modis_active_cells"]) > 0
        and np.median(paired_valid["viirs_active_cells"]) > 0
    ):
        global_alpha_cells, global_alpha_frp, global_raw_ratio = _safe_median_ratios(paired_valid)
    elif (
        len(coactive_all) >= 3
        and np.median(coactive_all["modis_active_cells"]) > 0
        and np.median(coactive_all["viirs_active_cells"]) > 0
    ):
        global_alpha_cells, global_alpha_frp, global_raw_ratio = _safe_median_ratios(coactive_all)
    else:
        has_global_modis = False
        global_alpha_cells = default_alpha_overall
        global_alpha_frp = default_alpha_overall
        global_raw_ratio = default_alpha_overall

    dn_fallbacks: dict[str, tuple[float, float, float]] = {}
    for dn in ("D", "N"):
        sub_dn = paired_valid[paired_valid["daynight"] == dn]
        sub_dn_pos = sub_dn[sub_dn["modis_active_cells"] > 0]
        if (
            len(sub_dn) >= 3
            and np.median(sub_dn["modis_active_cells"]) > 0
            and np.median(sub_dn["viirs_active_cells"]) > 0
        ):
            dn_fallbacks[dn] = _safe_median_ratios(sub_dn)
        elif (
            len(sub_dn_pos) >= 3
            and np.median(sub_dn_pos["modis_active_cells"]) > 0
            and np.median(sub_dn_pos["viirs_active_cells"]) > 0
        ):
            dn_fallbacks[dn] = _safe_median_ratios(sub_dn_pos)
        elif has_global_modis:
            dn_fallbacks[dn] = (global_alpha_cells, global_alpha_frp, global_raw_ratio)
        else:
            d_val = default_alpha_by_dn[dn]
            dn_fallbacks[dn] = (d_val, d_val, d_val)

    cal_rows: list[dict[str, object]] = []
    for month in range(1, 13):
        for dn in ("D", "N"):
            slice_df = paired_valid[
                (paired_valid["month"] == month) & (paired_valid["daynight"] == dn)
            ]
            if (
                len(slice_df) >= 3
                and np.median(slice_df["modis_active_cells"]) > 0
                and np.median(slice_df["viirs_active_cells"]) > 0
            ):
                m_cells = slice_df["modis_active_cells"].to_numpy(dtype=float)
                v_cells = slice_df["viirs_active_cells"].to_numpy(dtype=float)
                alpha_cells, alpha_frp, raw_ratio = _safe_median_ratios(slice_df)
                ci_low, ci_high = _bootstrap_median_ratio_ci(
                    m_cells, v_cells, n_bootstrap=n_bootstrap, seed=seed + month * 10 + (1 if dn == "N" else 0)
                )
                used_fallback = False
            else:
                alpha_cells, alpha_frp, raw_ratio = dn_fallbacks[dn]
                ci_low, ci_high = alpha_cells, alpha_cells
                used_fallback = True

            cal_rows.append(
                {
                    "month": month,
                    "daynight": dn,
                    "alpha_cells": round(alpha_cells, 4),
                    "alpha_ci_low": round(ci_low, 4),
                    "alpha_ci_high": round(ci_high, 4),
                    "alpha_frp": round(alpha_frp, 4),
                    "raw_point_ratio": round(raw_ratio, 4),
                    "overlap_samples": int(len(slice_df)),
                    "used_fallback": used_fallback,
                }
            )

    calibration_df = pd.DataFrame(cal_rows)

    # Monthly (year, month) totals across day+night for scatter plot & correlation metrics
    if paired.empty:
        overlap_monthly = pd.DataFrame(
            columns=[
                "year",
                "month",
                "modis_active_cells",
                "viirs_active_cells",
                "modis_raw_points",
                "viirs_raw_points",
                "modis_frp_sum",
                "viirs_frp_sum",
            ]
        )
    else:
        overlap_monthly = paired.groupby(["year", "month"], as_index=False).sum(numeric_only=True)

    valid_corr = overlap_monthly[
        (overlap_monthly["modis_active_cells"] > 0) & (overlap_monthly["viirs_active_cells"] > 0)
    ]
    if len(valid_corr) >= 3:
        x = valid_corr["viirs_active_cells"].to_numpy(dtype=float)
        y = valid_corr["modis_active_cells"].to_numpy(dtype=float)
        reg = stats.linregress(x, y)
        y_pred = reg.slope * x + reg.intercept
        corr_r2 = float(r2_score(y, y_pred))
        corr_stats = {
            "overlap_pearson_r": round(float(reg.rvalue), 4),
            "overlap_r2": round(corr_r2, 4),
            "ols_slope": round(float(reg.slope), 4),
            "ols_intercept": round(float(reg.intercept), 4),
        }
    else:
        corr_stats = {
            "overlap_pearson_r": 0.0,
            "overlap_r2": 0.0,
            "ols_slope": global_alpha_cells,
            "ols_intercept": 0.0,
        }

    return calibration_df, overlap_monthly, corr_stats


def build_harmonized_series(
    gridded_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    combination_mode: str = "calibrated_viirs",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 4b: Apply alpha(m, daynight) and assemble continuous monthly & cell-level series."""
    if gridded_df.empty:
        empty_monthly = pd.DataFrame(
            columns=[
                "year",
                "month",
                "date_label",
                "modis_raw_points",
                "viirs_raw_points",
                "raw_total_points",
                "modis_active_cells",
                "viirs_active_cells",
                "gridded_total_cells",
                "viirs_calibrated_cells",
                "ensemble_hotspots",
                "harmonized_hotspots",
                "modis_frp_sum",
                "viirs_frp_sum",
                "raw_total_frp",
                "viirs_calibrated_frp",
                "harmonized_frp",
                "primary_source",
            ]
        )
        return empty_monthly, gridded_df.copy()

    # Attach monthly & daynight alpha weights to gridded VIIRS cells
    cells = gridded_df.merge(
        calibration_df[["month", "daynight", "alpha_cells", "alpha_frp"]],
        on=["month", "daynight"],
        how="left",
    )
    cells["alpha_cells"] = cells["alpha_cells"].fillna(1.0)
    cells["alpha_frp"] = cells["alpha_frp"].fillna(1.0)
    is_viirs = cells["sensor_group"] == "VIIRS"
    cells["applied_alpha_cells"] = np.where(is_viirs, cells["alpha_cells"], 1.0)
    cells["applied_alpha_frp"] = np.where(is_viirs, cells["alpha_frp"], 1.0)
    cells["calibrated_cell_count"] = np.round(cells["active_cell_count"] * cells["applied_alpha_cells"], 4)
    cells["calibrated_frp"] = np.round(cells["frp_sum"] * cells["applied_alpha_frp"], 3)

    # Monthly aggregation per sensor
    m_sub = cells[cells["sensor_group"] == "MODIS"].groupby(["year", "month"], as_index=False).agg(
        modis_raw_points=("raw_point_count", "sum"),
        modis_active_cells=("active_cell_count", "sum"),
        modis_frp_sum=("frp_sum", "sum"),
    )
    v_sub = cells[cells["sensor_group"] == "VIIRS"].groupby(["year", "month"], as_index=False).agg(
        viirs_raw_points=("raw_point_count", "sum"),
        viirs_active_cells=("active_cell_count", "sum"),
        viirs_frp_sum=("frp_sum", "sum"),
        viirs_calibrated_cells=("calibrated_cell_count", "sum"),
        viirs_calibrated_frp=("calibrated_frp", "sum"),
    )

    min_year, max_year = int(cells["year"].min()), int(cells["year"].max())
    full_grid = pd.MultiIndex.from_product(
        [range(min_year, max_year + 1), range(1, 13)],
        names=["year", "month"],
    ).to_frame(index=False)

    monthly = (
        full_grid.merge(m_sub, on=["year", "month"], how="left")
        .merge(v_sub, on=["year", "month"], how="left")
        .fillna(0.0)
    )

    monthly["date_label"] = [f"{int(y):04d}-{int(m):02d}" for y, m in zip(monthly["year"], monthly["month"])]
    monthly["raw_total_points"] = monthly["modis_raw_points"] + monthly["viirs_raw_points"]
    monthly["gridded_total_cells"] = monthly["modis_active_cells"] + monthly["viirs_active_cells"]
    monthly["raw_total_frp"] = monthly["modis_frp_sum"] + monthly["viirs_frp_sum"]

    has_viirs = monthly["viirs_active_cells"] > 0
    has_modis = monthly["modis_active_cells"] > 0
    is_post_transition = (monthly["year"] > 2012) | ((monthly["year"] == 2012) & (monthly["month"] >= 3))

    monthly["ensemble_hotspots"] = np.where(
        has_modis & has_viirs,
        0.5 * (monthly["modis_active_cells"] + monthly["viirs_calibrated_cells"]),
        np.where(has_viirs, monthly["viirs_calibrated_cells"], monthly["modis_active_cells"]),
    )
    ensemble_frp = np.where(
        has_modis & has_viirs,
        0.5 * (monthly["modis_frp_sum"] + monthly["viirs_calibrated_frp"]),
        np.where(has_viirs, monthly["viirs_calibrated_frp"], monthly["modis_frp_sum"]),
    )

    if combination_mode == "ensemble":
        monthly["harmonized_hotspots"] = np.round(monthly["ensemble_hotspots"], 2)
        monthly["harmonized_frp"] = np.round(ensemble_frp, 2)
    else:
        use_viirs = is_post_transition & has_viirs
        monthly["harmonized_hotspots"] = np.round(
            np.where(use_viirs, monthly["viirs_calibrated_cells"], monthly["modis_active_cells"]),
            2,
        )
        monthly["harmonized_frp"] = np.round(
            np.where(use_viirs, monthly["viirs_calibrated_frp"], monthly["modis_frp_sum"]),
            2,
        )

    monthly["primary_source"] = np.where(
        ~is_post_transition | (~has_viirs),
        "MODIS",
        np.where(
            (combination_mode == "ensemble") & has_modis,
            "MODIS+VIIRS Ensemble",
            "VIIRS Calibrated",
        ),
    )

    # Harmonized cell-level table for Geospatial Explorer & CSV export
    if combination_mode == "ensemble":
        harmonized_cells = cells.copy()
        harmonized_cells["period_source"] = np.where(
            harmonized_cells["sensor_group"] == "MODIS",
            "MODIS",
            "VIIRS Calibrated",
        )
    else:
        is_pre_viirs = cells["acq_date"] < VIIRS_TRANSITION_DATE
        # Prefer MODIS before 2012-03-01 and calibrated VIIRS from 2012-03-01 onward
        viirs_months = set(zip(monthly.loc[has_viirs, "year"], monthly.loc[has_viirs, "month"]))
        in_viirs_month = [
            (int(y), int(m)) in viirs_months
            for y, m in zip(cells["year"], cells["month"])
        ]
        keep_mask = (
            (is_pre_viirs & (cells["sensor_group"] == "MODIS"))
            | (~is_pre_viirs & (cells["sensor_group"] == "VIIRS"))
            | (~is_pre_viirs & ~pd.Series(in_viirs_month, index=cells.index) & (cells["sensor_group"] == "MODIS"))
        )
        harmonized_cells = cells.loc[keep_mask].copy().reset_index(drop=True)
        harmonized_cells["period_source"] = np.where(
            harmonized_cells["sensor_group"] == "MODIS",
            "MODIS (Pre-VIIRS)",
            "VIIRS Calibrated",
        )

    return monthly, harmonized_cells


def evaluate_2012_transition(
    monthly_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    corr_stats: dict[str, float],
    reference_metrics: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Quantify how much of the artificial 2012 sensor step jump was removed.

    For short operational windows (e.g. 1–30 day NRT API queries without both
    pre-2012 MODIS and post-2012 VIIRS history), optionally carries forward
    historical regional transition benchmarks via ``reference_metrics`` while
    reporting live window totals and flagging ``used_reference_calibration``.
    """
    annual = monthly_df.groupby("year", as_index=False).sum(numeric_only=True)
    pre = annual[annual["year"] <= 2011]
    post = annual[annual["year"] >= 2013]

    day_alpha = float(calibration_df.loc[calibration_df["daynight"] == "D", "alpha_cells"].mean())
    night_alpha = float(calibration_df.loc[calibration_df["daynight"] == "N", "alpha_cells"].mean())
    overall_alpha = float(calibration_df["alpha_cells"].mean())

    live_raw_points = float(monthly_df["raw_total_points"].sum()) if not monthly_df.empty else 0.0
    live_gridded_cells = float(monthly_df["gridded_total_cells"].sum()) if not monthly_df.empty else 0.0
    live_harmonized = float(monthly_df["harmonized_hotspots"].sum()) if not monthly_df.empty else 0.0

    is_short_window = pre.empty or post.empty or (monthly_df["year"].nunique() < 2)
    if is_short_window and reference_metrics:
        return {
            "pre2012_raw_annual_mean": float(reference_metrics.get("pre2012_raw_annual_mean", 0.0)),
            "post2012_raw_combined_annual_mean": float(
                reference_metrics.get("post2012_raw_combined_annual_mean", 0.0)
            ),
            "raw_combined_jump_pct": float(reference_metrics.get("raw_combined_jump_pct", 0.0)),
            "post2012_raw_viirs_annual_mean": float(
                reference_metrics.get("post2012_raw_viirs_annual_mean", 0.0)
            ),
            "raw_viirs_jump_pct": float(reference_metrics.get("raw_viirs_jump_pct", 0.0)),
            "pre2012_gridded_annual_mean": float(
                reference_metrics.get("pre2012_gridded_annual_mean", 0.0)
            ),
            "post2012_gridded_viirs_annual_mean": float(
                reference_metrics.get("post2012_gridded_viirs_annual_mean", 0.0)
            ),
            "gridded_uncalibrated_jump_pct": float(
                reference_metrics.get("gridded_uncalibrated_jump_pct", 0.0)
            ),
            "post2012_harmonized_annual_mean": float(
                reference_metrics.get("post2012_harmonized_annual_mean", 0.0)
            ),
            "harmonized_diff_pct": float(reference_metrics.get("harmonized_diff_pct", 0.0)),
            "overlap_sensor_residual_bias_pct": float(
                reference_metrics.get("overlap_sensor_residual_bias_pct", 0.0)
            ),
            "jump_reduction_pct": float(reference_metrics.get("jump_reduction_pct", 0.0)),
            "mean_alpha_day": round(day_alpha, 4),
            "mean_alpha_night": round(night_alpha, 4),
            "mean_alpha_overall": round(overall_alpha, 4),
            "overlap_pearson_r": float(reference_metrics.get("overlap_pearson_r", 0.0)),
            "overlap_r2": float(reference_metrics.get("overlap_r2", 0.0)),
            "ols_slope": float(reference_metrics.get("ols_slope", overall_alpha)),
            "ols_intercept": float(reference_metrics.get("ols_intercept", 0.0)),
            "used_reference_calibration": True,
            "live_raw_points": round(live_raw_points, 1),
            "live_gridded_cells": round(live_gridded_cells, 1),
            "live_harmonized_hotspots": round(live_harmonized, 1),
        }

    pre_raw_mean = float(pre["raw_total_points"].mean()) if not pre.empty else 0.0
    post_raw_combined_mean = float(post["raw_total_points"].mean()) if not post.empty else 0.0
    post_raw_viirs_mean = float(post["viirs_raw_points"].mean()) if not post.empty else 0.0
    post_gridded_viirs_mean = float(post["viirs_active_cells"].mean()) if not post.empty else 0.0

    pre_harm_mean = float(pre["harmonized_hotspots"].mean()) if not pre.empty else 0.0
    post_harm_mean = float(post["harmonized_hotspots"].mean()) if not post.empty else 0.0

    def _pct_jump(after_val: float, before_val: float) -> float:
        if before_val <= 0:
            return 0.0
        return ((after_val - before_val) / before_val) * 100.0

    raw_combined_jump_pct = _pct_jump(post_raw_combined_mean, pre_raw_mean)
    raw_viirs_jump_pct = _pct_jump(post_raw_viirs_mean, pre_raw_mean)
    gridded_uncal_jump_pct = _pct_jump(post_gridded_viirs_mean, pre_harm_mean)
    harmonized_diff_pct = _pct_jump(post_harm_mean, pre_harm_mean)

    # Sensor-only residual bias during overlap years (2012–2025): calibrated VIIRS vs MODIS
    ov = annual[(annual["year"] >= 2012) & (annual["year"] <= 2025)]
    ov_modis = float(ov["modis_active_cells"].sum()) if not ov.empty else 0.0
    ov_viirs_cal = float(ov["viirs_calibrated_cells"].sum()) if not ov.empty else 0.0
    overlap_sensor_residual_bias_pct = _pct_jump(ov_viirs_cal, ov_modis)

    jump_reduction_pct = 0.0
    if abs(raw_combined_jump_pct) > 1e-6:
        jump_reduction_pct = float(
            np.clip((1.0 - abs(harmonized_diff_pct) / abs(raw_combined_jump_pct)) * 100.0, 0.0, 100.0)
        )

    return {
        "pre2012_raw_annual_mean": round(pre_raw_mean, 1),
        "post2012_raw_combined_annual_mean": round(post_raw_combined_mean, 1),
        "raw_combined_jump_pct": round(raw_combined_jump_pct, 2),
        "post2012_raw_viirs_annual_mean": round(post_raw_viirs_mean, 1),
        "raw_viirs_jump_pct": round(raw_viirs_jump_pct, 2),
        "pre2012_gridded_annual_mean": round(pre_harm_mean, 1),
        "post2012_gridded_viirs_annual_mean": round(post_gridded_viirs_mean, 1),
        "gridded_uncalibrated_jump_pct": round(gridded_uncal_jump_pct, 2),
        "post2012_harmonized_annual_mean": round(post_harm_mean, 1),
        "harmonized_diff_pct": round(harmonized_diff_pct, 2),
        "overlap_sensor_residual_bias_pct": round(overlap_sensor_residual_bias_pct, 2),
        "jump_reduction_pct": round(jump_reduction_pct, 2),
        "mean_alpha_day": round(day_alpha, 4),
        "mean_alpha_night": round(night_alpha, 4),
        "mean_alpha_overall": round(overall_alpha, 4),
        "used_reference_calibration": False,
        "live_raw_points": round(live_raw_points, 1),
        "live_gridded_cells": round(live_gridded_cells, 1),
        "live_harmonized_hotspots": round(live_harmonized, 1),
        **corr_stats,
    }


class HarmonizationPipeline:
    """End-to-end 4-stage harmonization orchestrator for MODIS and VIIRS active fires."""

    def __init__(
        self,
        modis_min_conf: int = DEFAULT_MODIS_MIN_CONFIDENCE,
        viirs_allowed_conf: Optional[Iterable[str]] = None,
        grid_step_deg: float = GRID_STEP_DEG,
        overlap_years: tuple[int, int] = OVERLAP_YEARS,
        combination_mode: str = "calibrated_viirs",
        daynight_filter: str = "All",
    ) -> None:
        self.modis_min_conf = modis_min_conf
        self.viirs_allowed_conf = list(
            viirs_allowed_conf if viirs_allowed_conf is not None else DEFAULT_VIIRS_CONFIDENCE
        )
        self.grid_step_deg = grid_step_deg
        self.overlap_years = overlap_years
        self.combination_mode = combination_mode
        self.daynight_filter = daynight_filter

    def run(
        self,
        modis_df: pd.DataFrame,
        viirs_df: pd.DataFrame,
        external_calibration_df: Optional[pd.DataFrame] = None,
        external_overlap_df: Optional[pd.DataFrame] = None,
        reference_transition_metrics: Optional[dict[str, float]] = None,
    ) -> HarmonizationResult:
        """Execute all 4 harmonization stages and return structured outputs."""
        unified = unify_and_filter_sensors(
            modis_df=modis_df,
            viirs_df=viirs_df,
            modis_min_conf=self.modis_min_conf,
            viirs_allowed_conf=self.viirs_allowed_conf,
            daynight_filter=self.daynight_filter,
        )
        normalized = normalize_pixel_geometry(unified)
        gridded = grid_spatiotemporal_clusters(normalized, grid_step=self.grid_step_deg)
        calibration_df, overlap_monthly, corr_stats = compute_overlap_calibration(
            gridded,
            overlap_years=self.overlap_years,
        )
        if external_calibration_df is not None and not external_calibration_df.empty:
            if calibration_df["used_fallback"].all() or gridded["year"].nunique() < 2:
                calibration_df = external_calibration_df.copy()
                if external_overlap_df is not None and len(overlap_monthly) < 3:
                    overlap_monthly = external_overlap_df.copy()
        monthly_df, harmonized_cells = build_harmonized_series(
            gridded_df=gridded,
            calibration_df=calibration_df,
            combination_mode=self.combination_mode,
        )
        transition_metrics = evaluate_2012_transition(
            monthly_df=monthly_df,
            calibration_df=calibration_df,
            corr_stats=corr_stats,
            reference_metrics=reference_transition_metrics,
        )
        return HarmonizationResult(
            filtered_points_df=normalized,
            gridded_df=gridded,
            calibration_df=calibration_df,
            monthly_df=monthly_df,
            harmonized_cells_df=harmonized_cells,
            overlap_monthly_df=overlap_monthly,
            transition_metrics=transition_metrics,
        )
