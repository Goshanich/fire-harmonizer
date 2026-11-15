"""4-stage MODIS and VIIRS active fire harmonization engine.

Implements NASA FIRMS schema unification & confidence filtering, pixel swath-area
and FRP normalization, 0.01 deg (~1 km) spatiotemporal gridding, and overlap
cross-calibration (2012–2025) with bootstrap 95% confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Optional

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
            conf_num = pd.to_numeric(
                m.get("confidence", pd.Series(50.0, index=m.index)), errors="coerce"
            ).fillna(0.0)
            m = m[conf_num >= float(modis_min_conf)].copy()
            if not m.empty:
                conf_kept = pd.to_numeric(
                    m.get("confidence", pd.Series(50.0, index=m.index)), errors="coerce"
                ).fillna(50.0)
                bright_col = "brightness" if "brightness" in m.columns else "bright_ti4"
                m_out = pd.DataFrame(
                    {
                        "latitude": pd.to_numeric(m["latitude"], errors="coerce"),
                        "longitude": pd.to_numeric(m["longitude"], errors="coerce"),
                        "acq_date": m["acq_date"].astype(str),
                        "acq_time": m.get("acq_time", pd.Series("1200", index=m.index)).astype(str).str.zfill(4),
                        "daynight": m.get("daynight", pd.Series("D", index=m.index)).astype(str).str.upper(),
                        "scan": pd.to_numeric(m.get("scan", pd.Series(1.0, index=m.index)), errors="coerce").fillna(1.0),
                        "track": pd.to_numeric(m.get("track", pd.Series(1.0, index=m.index)), errors="coerce").fillna(1.0),
                        "frp": pd.to_numeric(m.get("frp", pd.Series(0.0, index=m.index)), errors="coerce").fillna(0.0).clip(lower=0.0),
                        "brightness_k": pd.to_numeric(m.get(bright_col, pd.Series(np.nan, index=m.index)), errors="coerce"),
                        "confidence_raw": conf_kept.astype(int).astype(str),
                        "confidence_weight": np.clip(conf_kept / 100.0, 0.0, 1.0),
                        "sensor_group": "MODIS",
                        "satellite": m.get("satellite", pd.Series("Terra/Aqua", index=m.index)).astype(str),
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
                        "acq_time": v.get("acq_time", pd.Series("1200", index=v.index)).astype(str).str.zfill(4),
                        "daynight": v.get("daynight", pd.Series("D", index=v.index)).astype(str).str.upper(),
                        "scan": pd.to_numeric(v.get("scan", pd.Series(0.45, index=v.index)), errors="coerce").fillna(0.45),
                        "track": pd.to_numeric(v.get("track", pd.Series(0.45, index=v.index)), errors="coerce").fillna(0.45),
                        "frp": pd.to_numeric(v.get("frp", pd.Series(0.0, index=v.index)), errors="coerce").fillna(0.0).clip(lower=0.0),
                        "brightness_k": pd.to_numeric(v.get(bright_col, pd.Series(np.nan, index=v.index)), errors="coerce"),
                        "confidence_raw": conf_kept,
                        "confidence_weight": conf_kept.map(VIIRS_CONFIDENCE_WEIGHTS).fillna(0.75),
                        "sensor_group": "VIIRS",
                        "satellite": v.get("satellite", pd.Series("SNPP", index=v.index)).astype(str),
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


def _bootstrap_sum_ratio_ci(
    modis_vals: np.ndarray,
    viirs_vals: np.ndarray,
    base_alpha: float,
    shrink_weight: float,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% paired bootstrap CI for shrunk sum-ratio Sum(MODIS) / Sum(VIIRS).

    Resamples paired ``(year, month)`` vectors together with the same index draw so
    season/year covariance between MODIS and VIIRS is preserved.
    """
    n = len(modis_vals)
    if n < 3 or float(np.sum(viirs_vals)) <= 0.0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    b_modis = modis_vals[idx].sum(axis=1)
    b_viirs = viirs_vals[idx].sum(axis=1)
    valid = b_viirs > 0.0
    if int(np.sum(valid)) < 5:
        return np.nan, np.nan

    ratios = b_modis[valid] / b_viirs[valid]
    shrunk = shrink_weight * ratios + (1.0 - shrink_weight) * base_alpha
    low = float(np.percentile(shrunk, 2.5))
    high = float(np.percentile(shrunk, 97.5))
    return low, high


def compute_overlap_calibration(
    gridded_df: pd.DataFrame,
    overlap_years: tuple[int, int] = OVERLAP_YEARS,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Step 4a: Derive sum-ratio alpha(dn, m) factors with empirical-Bayes shrinkage on 2012–2025 overlap."""
    y_start, y_end = overlap_years
    if gridded_df.empty:
        overlap_raw = gridded_df.copy()
    else:
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
    total_m_cells = float(paired["modis_active_cells"].sum()) if not paired.empty else 0.0
    total_v_cells = float(paired["viirs_active_cells"].sum()) if not paired.empty else 0.0
    has_valid_overlap = bool(not paired.empty and total_m_cells > 0.0 and total_v_cells > 0.0)

    if not has_valid_overlap:
        cal_rows = [
            {
                "month": month,
                "daynight": dn,
                "alpha_cells": 1.0,
                "alpha_ci_low": np.nan,
                "alpha_ci_high": np.nan,
                "alpha_frp": 1.0,
                "raw_point_ratio": 1.0,
                "overlap_samples": 0,
                "used_fallback": True,
                "calibration_status": "no_data",
            }
            for month in range(1, 13)
            for dn in ("D", "N")
        ]
        calibration_df = pd.DataFrame(cal_rows)
        empty_ov = pd.DataFrame(
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
        return calibration_df, empty_ov, {
            "overlap_pearson_r": 0.0,
            "overlap_r2": 0.0,
            "ols_slope": 1.0,
            "ols_intercept": 0.0,
            "overlap_alpha_day": 1.0,
            "overlap_alpha_night": 1.0,
            "overlap_alpha_overall": 1.0,
            "has_valid_overlap": False,
        }

    # Global & Day/Night sum ratios: alpha(dn) = Sum(MODIS_cells) / Sum(VIIRS_cells)
    global_alpha_cells = total_m_cells / total_v_cells
    total_m_frp = float(paired["modis_frp_sum"].sum())
    total_v_frp = float(paired["viirs_frp_sum"].sum())
    global_alpha_frp = (
        total_m_frp / total_v_frp if (total_m_frp > 0.0 and total_v_frp > 0.0) else global_alpha_cells
    )
    total_m_raw = float(paired["modis_raw_points"].sum())
    total_v_raw = float(paired["viirs_raw_points"].sum())
    global_raw_ratio = (
        total_m_raw / total_v_raw if (total_m_raw > 0.0 and total_v_raw > 0.0) else global_alpha_cells
    )

    cal_rows: list[dict[str, object]] = []
    dn_headline_alphas: dict[str, float] = {}

    for dn in ("D", "N"):
        sub_dn = paired[paired["daynight"] == dn]
        sum_m_dn = float(sub_dn["modis_active_cells"].sum())
        sum_v_dn = float(sub_dn["viirs_active_cells"].sum())
        base_alpha_dn = (
            sum_m_dn / sum_v_dn if (sum_m_dn > 0.0 and sum_v_dn > 0.0) else global_alpha_cells
        )
        dn_headline_alphas[dn] = base_alpha_dn

        sum_m_frp_dn = float(sub_dn["modis_frp_sum"].sum())
        sum_v_frp_dn = float(sub_dn["viirs_frp_sum"].sum())
        base_frp_dn = (
            sum_m_frp_dn / sum_v_frp_dn
            if (sum_m_frp_dn > 0.0 and sum_v_frp_dn > 0.0)
            else global_alpha_frp
        )
        sum_m_raw_dn = float(sub_dn["modis_raw_points"].sum())
        sum_v_raw_dn = float(sub_dn["viirs_raw_points"].sum())
        base_raw_dn = (
            sum_m_raw_dn / sum_v_raw_dn
            if (sum_m_raw_dn > 0.0 and sum_v_raw_dn > 0.0)
            else global_raw_ratio
        )

        dn_month_rows: list[dict[str, Any]] = []
        for month in range(1, 13):
            slice_df = sub_dn[sub_dn["month"] == month]
            paired_both = slice_df[
                (slice_df["modis_active_cells"] > 0.0) & (slice_df["viirs_active_cells"] > 0.0)
            ]
            n_paired = int(len(paired_both))
            m_sum = float(slice_df["modis_active_cells"].sum())
            v_sum = float(slice_df["viirs_active_cells"].sum())
            m_frp = float(slice_df["modis_frp_sum"].sum())
            v_frp = float(slice_df["viirs_frp_sum"].sum())
            m_raw = float(slice_df["modis_raw_points"].sum())
            v_raw = float(slice_df["viirs_raw_points"].sum())

            if n_paired < 3 or m_sum <= 0.0 or v_sum <= 0.0:
                dn_month_rows.append(
                    {
                        "month": month,
                        "daynight": dn,
                        "alpha_tilde": base_alpha_dn,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "frp_tilde": base_frp_dn,
                        "raw_ratio": base_raw_dn,
                        "v_sum": v_sum,
                        "v_frp": v_frp,
                        "overlap_samples": n_paired,
                        "used_fallback": True,
                        "calibration_status": "no_data",
                    }
                )
            else:
                month_ratio = m_sum / v_sum
                # Empirical-Bayes shrinkage toward the day/night sum ratio
                shrink_w = min(0.35, n_paired / (n_paired + 12.0))
                alpha_tilde = shrink_w * month_ratio + (1.0 - shrink_w) * base_alpha_dn

                ci_low, ci_high = _bootstrap_sum_ratio_ci(
                    slice_df["modis_active_cells"].to_numpy(dtype=float),
                    slice_df["viirs_active_cells"].to_numpy(dtype=float),
                    base_alpha=base_alpha_dn,
                    shrink_weight=shrink_w,
                    n_bootstrap=n_bootstrap,
                    seed=seed + month * 10 + (1 if dn == "N" else 0),
                )
                if np.isfinite(ci_low) and np.isfinite(ci_high):
                    ci_low = min(ci_low, alpha_tilde)
                    ci_high = max(ci_high, alpha_tilde)

                # FRP ratio: use base_frp_dn when sparse (<5 paired months) and clamp extreme outliers
                if n_paired < 5 or m_frp <= 0.0 or v_frp <= 0.0:
                    frp_tilde = base_frp_dn
                else:
                    raw_frp_ratio = float(
                        np.clip(m_frp / v_frp, 0.35 * base_frp_dn, 1.75 * base_frp_dn)
                    )
                    w_frp = min(0.25, n_paired / (n_paired + 15.0))
                    frp_tilde = w_frp * raw_frp_ratio + (1.0 - w_frp) * base_frp_dn

                raw_ratio = (m_raw / v_raw) if (m_raw > 0.0 and v_raw > 0.0) else base_raw_dn
                dn_month_rows.append(
                    {
                        "month": month,
                        "daynight": dn,
                        "alpha_tilde": alpha_tilde,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "frp_tilde": frp_tilde,
                        "raw_ratio": raw_ratio,
                        "v_sum": v_sum,
                        "v_frp": v_frp,
                        "overlap_samples": n_paired,
                        "used_fallback": False,
                        "calibration_status": "calibrated",
                    }
                )

        # Normalize shrunk monthly weights so Sum(alpha_final * VIIRS_m) == Sum(MODIS) per day/night
        pred_v_dn = sum(r["alpha_tilde"] * r["v_sum"] for r in dn_month_rows)
        norm_cells = (sum_m_dn / pred_v_dn) if (pred_v_dn > 0.0 and sum_m_dn > 0.0) else 1.0

        pred_frp_dn = sum(r["frp_tilde"] * r["v_frp"] for r in dn_month_rows)
        norm_frp = (
            sum_m_frp_dn / pred_frp_dn if (pred_frp_dn > 0.0 and sum_m_frp_dn > 0.0) else 1.0
        )

        for r in dn_month_rows:
            a_val = float(r["alpha_tilde"] * norm_cells)
            ci_l = float(r["ci_low"] * norm_cells) if np.isfinite(r["ci_low"]) else np.nan
            ci_h = float(r["ci_high"] * norm_cells) if np.isfinite(r["ci_high"]) else np.nan
            f_val = float(r["frp_tilde"] * norm_frp)
            cal_rows.append(
                {
                    "month": int(r["month"]),
                    "daynight": str(r["daynight"]),
                    "alpha_cells": round(a_val, 5),
                    "alpha_ci_low": round(ci_l, 5) if np.isfinite(ci_l) else np.nan,
                    "alpha_ci_high": round(ci_h, 5) if np.isfinite(ci_h) else np.nan,
                    "alpha_frp": round(f_val, 5),
                    "raw_point_ratio": round(float(r["raw_ratio"]), 4),
                    "overlap_samples": int(r["overlap_samples"]),
                    "used_fallback": bool(r["used_fallback"]),
                    "calibration_status": str(r["calibration_status"]),
                }
            )

    calibration_df = (
        pd.DataFrame(cal_rows).sort_values(["month", "daynight"]).reset_index(drop=True)
    )

    overlap_monthly = paired.groupby(["year", "month"], as_index=False).sum(numeric_only=True)

    valid_corr = overlap_monthly[
        (overlap_monthly["modis_active_cells"] > 0) & (overlap_monthly["viirs_active_cells"] > 0)
    ]
    if len(valid_corr) >= 3:
        x = valid_corr["viirs_active_cells"].to_numpy(dtype=float)
        y = valid_corr["modis_active_cells"].to_numpy(dtype=float)
        if float(np.std(x)) > 1e-9 and float(np.std(y)) > 1e-9:
            reg = stats.linregress(x, y)
            y_pred = reg.slope * x + reg.intercept
            corr_r2 = float(r2_score(y, y_pred))
            rval = float(reg.rvalue)
            slope_val = float(reg.slope)
            intercept_val = float(reg.intercept)
        else:
            corr_r2 = 0.0
            rval = 0.0
            slope_val = float(global_alpha_cells)
            intercept_val = 0.0
        corr_stats = {
            "overlap_pearson_r": round(rval, 4),
            "overlap_r2": round(corr_r2, 4),
            "ols_slope": round(slope_val, 4),
            "ols_intercept": round(intercept_val, 4),
            "overlap_alpha_day": round(float(dn_headline_alphas.get("D", global_alpha_cells)), 4),
            "overlap_alpha_night": round(float(dn_headline_alphas.get("N", global_alpha_cells)), 4),
            "overlap_alpha_overall": round(float(global_alpha_cells), 4),
            "has_valid_overlap": True,
        }
    else:
        corr_stats = {
            "overlap_pearson_r": 0.0,
            "overlap_r2": 0.0,
            "ols_slope": round(float(global_alpha_cells), 4),
            "ols_intercept": 0.0,
            "overlap_alpha_day": round(float(dn_headline_alphas.get("D", global_alpha_cells)), 4),
            "overlap_alpha_night": round(float(dn_headline_alphas.get("N", global_alpha_cells)), 4),
            "overlap_alpha_overall": round(float(global_alpha_cells), 4),
            "has_valid_overlap": bool(not bool(calibration_df["used_fallback"].all())),
        }

    return calibration_df, overlap_monthly, corr_stats


def build_in_window_calibration(
    gridded_df: pd.DataFrame,
    min_cells_threshold: int = 30,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive honest in-window MODIS-equivalent scaling factors without invented constants.

    Populates all 12 calendar months x ('D', 'N') so multi-month short windows
    never leave uncovered months with silent uncalibrated values. Rows without sufficient
    paired data receive NaN CI bounds and ``calibration_status='no_data'``.
    """
    cols = [
        "month",
        "daynight",
        "alpha_cells",
        "alpha_ci_low",
        "alpha_ci_high",
        "alpha_frp",
        "raw_point_ratio",
        "overlap_samples",
        "used_fallback",
        "calibration_status",
        "n_modis_cells",
        "n_viirs_cells",
    ]
    if gridded_df is None or gridded_df.empty:
        n_modis_cells = 0
        n_viirs_cells = 0
        modis_frp_sum = 0.0
        viirs_frp_sum = 0.0
        modis_raw_pts = 0.0
        viirs_raw_pts = 0.0
        m_sub = pd.DataFrame()
        v_sub = pd.DataFrame()
        work = pd.DataFrame()
    else:
        dn_norm = gridded_df["daynight"].astype(str).str.upper()
        work = gridded_df.assign(_dn_upper=dn_norm)
        m_sub = work[work["sensor_group"] == "MODIS"]
        v_sub = work[work["sensor_group"] == "VIIRS"]
        n_modis_cells = int(m_sub["active_cell_count"].sum())
        n_viirs_cells = int(v_sub["active_cell_count"].sum())
        modis_frp_sum = float(m_sub["frp_sum"].sum())
        viirs_frp_sum = float(v_sub["frp_sum"].sum())
        modis_raw_pts = float(m_sub["raw_point_count"].sum())
        viirs_raw_pts = float(v_sub["raw_point_count"].sum())

    if (
        n_modis_cells == 0
        or n_modis_cells < int(min_cells_threshold)
        or n_viirs_cells < int(min_cells_threshold)
    ):
        overall_alpha_cells = 1.0
        overall_alpha_frp = 1.0
        raw_point_ratio = 1.0
        dn_alpha_cells = {"D": 1.0, "N": 1.0}
        dn_alpha_frp = {"D": 1.0, "N": 1.0}
        dn_ci: dict[str, tuple[float, float]] = {
            "D": (np.nan, np.nan),
            "N": (np.nan, np.nan),
        }
        used_fallback = True
        low_sample_warning = True
        calibration_status = "no_data"
        mode_label = "VIIRS native (not harmonized to MODIS)"
    else:
        overall_alpha_cells = float(n_modis_cells / max(n_viirs_cells, 1))
        overall_alpha_frp = (
            float(modis_frp_sum / max(viirs_frp_sum, 1e-6))
            if (modis_frp_sum > 0.0 and viirs_frp_sum > 0.0)
            else overall_alpha_cells
        )
        raw_point_ratio = (
            float(modis_raw_pts / max(viirs_raw_pts, 1.0))
            if modis_raw_pts > 0 and viirs_raw_pts > 0
            else overall_alpha_cells
        )
        dn_alpha_cells = {}
        dn_alpha_frp = {}
        dn_ci = {}
        for dn in ("D", "N"):
            m_dn = m_sub[m_sub["_dn_upper"] == dn]
            v_dn = v_sub[v_sub["_dn_upper"] == dn]
            n_modis_dn = int(m_dn["active_cell_count"].sum())
            n_viirs_dn = int(v_dn["active_cell_count"].sum())
            if n_modis_dn >= 10 and n_viirs_dn >= 10:
                a_dn = float(n_modis_dn / n_viirs_dn)
                dn_alpha_cells[dn] = a_dn
                m_dn_frp = float(m_dn["frp_sum"].sum())
                v_dn_frp = float(v_dn["frp_sum"].sum())
                dn_alpha_frp[dn] = (
                    float(m_dn_frp / max(v_dn_frp, 1e-6))
                    if m_dn_frp > 0 and v_dn_frp > 0
                    else overall_alpha_frp
                )
                # Paired daily bootstrap CI when >=3 dates observed
                m_day = m_dn.groupby("acq_date")["active_cell_count"].sum()
                v_day = v_dn.groupby("acq_date")["active_cell_count"].sum()
                day_pair = pd.concat([m_day.rename("m"), v_day.rename("v")], axis=1).fillna(0.0)
                if len(day_pair) >= 3:
                    dn_ci[dn] = _bootstrap_sum_ratio_ci(
                        day_pair["m"].to_numpy(dtype=float),
                        day_pair["v"].to_numpy(dtype=float),
                        base_alpha=a_dn,
                        shrink_weight=1.0,
                        n_bootstrap=n_bootstrap,
                        seed=seed + (1 if dn == "N" else 0),
                    )
                else:
                    dn_ci[dn] = (np.nan, np.nan)
            else:
                dn_alpha_cells[dn] = overall_alpha_cells
                dn_alpha_frp[dn] = overall_alpha_frp
                dn_ci[dn] = (np.nan, np.nan)

        used_fallback = False
        low_sample_warning = False
        calibration_status = "calibrated"
        mode_label = (
            f"In-window MODIS-equivalent (from {n_modis_cells} MODIS / {n_viirs_cells} VIIRS cells)"
        )

    overlap_samples = int(min(n_modis_cells, n_viirs_cells))
    rows: list[dict[str, Any]] = []
    for month in range(1, 13):
        for dn in ("D", "N"):
            a_cells = round(float(dn_alpha_cells[dn]), 5)
            a_frp = round(float(dn_alpha_frp[dn]), 5)
            ci_l, ci_h = dn_ci[dn]
            rows.append(
                {
                    "month": month,
                    "daynight": dn,
                    "alpha_cells": a_cells,
                    "alpha_ci_low": round(float(ci_l), 5) if np.isfinite(ci_l) else np.nan,
                    "alpha_ci_high": round(float(ci_h), 5) if np.isfinite(ci_h) else np.nan,
                    "alpha_frp": a_frp,
                    "raw_point_ratio": round(float(raw_point_ratio), 4),
                    "overlap_samples": overlap_samples,
                    "used_fallback": used_fallback,
                    "calibration_status": calibration_status,
                    "n_modis_cells": n_modis_cells,
                    "n_viirs_cells": n_viirs_cells,
                }
            )

    calibration_df = pd.DataFrame(rows, columns=cols)
    meta: dict[str, Any] = {
        "n_modis_cells": n_modis_cells,
        "n_viirs_cells": n_viirs_cells,
        "modis_frp_total": round(modis_frp_sum, 3),
        "viirs_frp_total": round(viirs_frp_sum, 3),
        "overall_alpha_cells": float(overall_alpha_cells),
        "overall_alpha_frp": float(overall_alpha_frp),
        "alpha_day": float(dn_alpha_cells["D"]),
        "alpha_night": float(dn_alpha_cells["N"]),
        "used_fallback": used_fallback,
        "low_sample_warning": low_sample_warning,
        "mode_label": mode_label,
        "min_cells_threshold": int(min_cells_threshold),
    }
    return calibration_df, meta


def build_daily_sensor_breakdown(
    gridded_df: pd.DataFrame,
    harmonized_cells_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate daily active cells and FRP separately for MODIS and VIIRS."""
    cols = [
        "acq_date",
        "modis_raw_points",
        "modis_cells",
        "modis_frp",
        "viirs_raw_points",
        "viirs_cells",
        "viirs_frp",
        "raw_points",
        "gridded_cells",
        "harmonized_cells",
        "harmonized_frp",
    ]
    if gridded_df is None or gridded_df.empty:
        return pd.DataFrame(columns=cols)

    def _agg(sensor: str, prefix: str) -> pd.DataFrame:
        sub = gridded_df[gridded_df["sensor_group"] == sensor]
        if sub.empty:
            return pd.DataFrame(
                columns=[
                    "acq_date",
                    f"{prefix}_raw_points",
                    f"{prefix}_cells",
                    f"{prefix}_frp",
                ]
            )
        return sub.groupby("acq_date", as_index=False).agg(
            **{
                f"{prefix}_raw_points": ("raw_point_count", "sum"),
                f"{prefix}_cells": ("active_cell_count", "sum"),
                f"{prefix}_frp": ("frp_sum", "sum"),
            }
        )

    m_daily = _agg("MODIS", "modis")
    v_daily = _agg("VIIRS", "viirs")
    daily = pd.merge(m_daily, v_daily, on="acq_date", how="outer").fillna(0.0)
    daily["raw_points"] = daily["modis_raw_points"] + daily["viirs_raw_points"]
    daily["gridded_cells"] = daily["modis_cells"] + daily["viirs_cells"]

    if (
        harmonized_cells_df is not None
        and not harmonized_cells_df.empty
        and "calibrated_cell_count" in harmonized_cells_df.columns
    ):
        is_ensemble_df = (
            "period_source" in harmonized_cells_df.columns
            and "MODIS" in set(harmonized_cells_df["period_source"].astype(str))
        )
        if is_ensemble_df and "sensor_group" in harmonized_cells_df.columns:
            hm = (
                harmonized_cells_df[harmonized_cells_df["sensor_group"] == "MODIS"]
                .groupby("acq_date", as_index=False)
                .agg(_hm_c=("calibrated_cell_count", "sum"), _hm_f=("calibrated_frp", "sum"))
            )
            hv = (
                harmonized_cells_df[harmonized_cells_df["sensor_group"] == "VIIRS"]
                .groupby("acq_date", as_index=False)
                .agg(_hv_c=("calibrated_cell_count", "sum"), _hv_f=("calibrated_frp", "sum"))
            )
            daily = daily.merge(hm, on="acq_date", how="left").merge(hv, on="acq_date", how="left").fillna(0.0)
            both = (daily["_hm_c"] > 0) & (daily["_hv_c"] > 0)
            daily["harmonized_cells"] = np.where(
                both, 0.5 * (daily["_hm_c"] + daily["_hv_c"]), daily["_hm_c"] + daily["_hv_c"]
            )
            daily["harmonized_frp"] = np.where(
                both, 0.5 * (daily["_hm_f"] + daily["_hv_f"]), daily["_hm_f"] + daily["_hv_f"]
            )
        else:
            h_daily = harmonized_cells_df.groupby("acq_date", as_index=False).agg(
                harmonized_cells=("calibrated_cell_count", "sum"),
                harmonized_frp=("calibrated_frp", "sum"),
            )
            daily = daily.merge(h_daily, on="acq_date", how="outer").fillna(0.0)
    else:
        daily["harmonized_cells"] = daily["gridded_cells"]
        daily["harmonized_frp"] = 0.0

    return daily[cols].sort_values("acq_date").reset_index(drop=True)


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
                "data_status",
                "sensor_coverage",
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
    min_obs_ym = str(cells["acq_date"].min())[:7]
    max_obs_ym = str(cells["acq_date"].max())[:7]
    observed_years = set(cells["year"].astype(int).unique())

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
    is_observed = (
        (monthly["date_label"] >= min_obs_ym)
        & (monthly["date_label"] <= max_obs_ym)
        & monthly["year"].isin(observed_years)
    )
    monthly["data_status"] = np.where(is_observed, "observed", "no_data")

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
        harm_cells_vals = np.round(monthly["ensemble_hotspots"], 2)
        harm_frp_vals = np.round(ensemble_frp, 2)
    else:
        use_viirs = is_post_transition & has_viirs
        harm_cells_vals = np.round(
            np.where(use_viirs, monthly["viirs_calibrated_cells"], monthly["modis_active_cells"]),
            2,
        )
        harm_frp_vals = np.round(
            np.where(use_viirs, monthly["viirs_calibrated_frp"], monthly["modis_frp_sum"]),
            2,
        )

    monthly["harmonized_hotspots"] = np.where(is_observed, harm_cells_vals, np.nan)
    monthly["harmonized_frp"] = np.where(is_observed, harm_frp_vals, np.nan)
    monthly["ensemble_hotspots"] = np.where(is_observed, np.round(monthly["ensemble_hotspots"], 2), np.nan)

    is_terra_only = monthly["date_label"] < "2002-07"
    monthly["sensor_coverage"] = np.where(
        ~is_observed,
        "no_data",
        np.where(
            ~is_post_transition,
            np.where(is_terra_only, "MODIS_only (Terra only)", "MODIS_only (Terra+Aqua)"),
            np.where(
                has_viirs,
                "MODIS+VIIRS_SNPP",
                np.where(has_modis, "MODIS_only (VIIRS missing)", "Observed (0 detections)"),
            ),
        ),
    )

    monthly["primary_source"] = np.where(
        ~is_observed,
        "No Data",
        np.where(
            ~is_post_transition | (~has_viirs),
            "MODIS",
            np.where(
                (combination_mode == "ensemble") & has_modis,
                "MODIS+VIIRS Ensemble",
                "VIIRS Calibrated",
            ),
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


def _compute_holdout_and_anomaly_metrics(
    monthly_df: pd.DataFrame,
    gridded_df: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """Compute Out-of-Sample (2012–2018 train -> 2019–2025 test) bias and deseasonalized anomaly r."""
    holdout_bias_pct = 0.0
    deseasonalized_r = 0.0

    if not monthly_df.empty:
        obs_mask = (
            monthly_df["data_status"] == "observed"
            if "data_status" in monthly_df.columns
            else pd.Series(True, index=monthly_df.index)
        )
        ov_m = monthly_df[
            obs_mask & (monthly_df["date_label"] >= "2012-03") & (monthly_df["year"] <= 2025)
        ].copy()
        ov_active = ov_m[
            (ov_m["modis_active_cells"] > 0) | (ov_m["viirs_calibrated_cells"] > 0)
        ].copy()
        if (
            len(ov_active) >= 6
            and (ov_active["modis_active_cells"] > 0).sum() >= 3
            and (ov_active["viirs_calibrated_cells"] > 0).sum() >= 3
        ):
            m_anom = (
                ov_active["modis_active_cells"]
                - ov_active.groupby("month")["modis_active_cells"].transform("mean")
            ).to_numpy(dtype=float)
            v_anom = (
                ov_active["viirs_calibrated_cells"]
                - ov_active.groupby("month")["viirs_calibrated_cells"].transform("mean")
            ).to_numpy(dtype=float)
            if float(np.std(m_anom)) > 1e-6 and float(np.std(v_anom)) > 1e-6:
                deseasonalized_r = float(stats.pearsonr(m_anom, v_anom)[0])

    if gridded_df is not None and not gridded_df.empty:
        cal_train, _, _ = compute_overlap_calibration(
            gridded_df,
            overlap_years=(2012, 2018),
            n_bootstrap=40,
        )
        if not bool(cal_train["used_fallback"].all()):
            holdout = gridded_df[
                (gridded_df["year"] >= 2019) & (gridded_df["year"] <= 2025)
            ]
            m_hold_sum = float(
                holdout.loc[holdout["sensor_group"] == "MODIS", "active_cell_count"].sum()
            )
            v_hold = holdout[holdout["sensor_group"] == "VIIRS"].merge(
                cal_train[["month", "daynight", "alpha_cells"]],
                on=["month", "daynight"],
                how="left",
            )
            v_hold_cal_sum = float(
                (v_hold["active_cell_count"] * v_hold["alpha_cells"].fillna(1.0)).sum()
            )
            if m_hold_sum > 0 and v_hold_cal_sum > 0:
                holdout_bias_pct = ((v_hold_cal_sum - m_hold_sum) / m_hold_sum) * 100.0
    elif not monthly_df.empty:
        obs_mask = (
            monthly_df["data_status"] == "observed"
            if "data_status" in monthly_df.columns
            else pd.Series(True, index=monthly_df.index)
        )
        train_m = monthly_df[
            obs_mask
            & (monthly_df["date_label"] >= "2012-03")
            & (monthly_df["year"] <= 2018)
            & (monthly_df["modis_active_cells"] > 0)
            & (monthly_df["viirs_active_cells"] > 0)
        ]
        test_m = monthly_df[
            obs_mask & (monthly_df["year"] >= 2019) & (monthly_df["year"] <= 2025)
        ]
        if not train_m.empty and not test_m.empty:
            global_a = float(
                train_m["modis_active_cells"].sum()
                / max(float(train_m["viirs_active_cells"].sum()), 1e-6)
            )
            month_alphas: dict[int, float] = {}
            for m_num, grp in train_m.groupby("month"):
                sum_v = float(grp["viirs_active_cells"].sum())
                sum_m = float(grp["modis_active_cells"].sum())
                month_alphas[int(m_num)] = (
                    (0.35 * (sum_m / sum_v) + 0.65 * global_a)
                    if (sum_v > 0 and sum_m > 0 and len(grp) >= 3)
                    else global_a
                )
            alphas_series = test_m["month"].map(lambda m: month_alphas.get(int(m), global_a))
            v_hold_cal = float((test_m["viirs_active_cells"] * alphas_series).sum())
            m_hold = float(test_m["modis_active_cells"].sum())
            if m_hold > 0 and v_hold_cal > 0:
                holdout_bias_pct = ((v_hold_cal - m_hold) / m_hold) * 100.0

    return {
        "holdout_bias_pct": round(holdout_bias_pct, 2),
        "deseasonalized_anomaly_r": round(deseasonalized_r, 4),
    }


def evaluate_2012_transition(
    monthly_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    corr_stats: dict[str, float],
    reference_metrics: Optional[dict[str, float]] = None,
    gridded_df: Optional[pd.DataFrame] = None,
    pre2012_start_year: int = 2003,
) -> dict[str, Any]:
    """Evaluate harmonization accuracy, overlap residual bias, holdout, and natural variability.

    Uses ``pre2012_start_year`` (default 2003, excluding 2001–2002 Terra-only years)
    when comparing pre-2012 and post-2012 baselines, and benchmarks harmonized differences
    against the MODIS-only natural variability signal.
    """
    obs_monthly = (
        monthly_df[monthly_df["data_status"] == "observed"].copy()
        if (not monthly_df.empty and "data_status" in monthly_df.columns)
        else monthly_df.copy()
    )
    annual = obs_monthly.groupby("year", as_index=False).sum(numeric_only=True) if not obs_monthly.empty else obs_monthly
    current_year = date.today().year
    annual_complete = annual[annual["year"] < current_year] if not annual.empty else annual

    pre = (
        annual_complete[
            (annual_complete["year"] >= int(pre2012_start_year)) & (annual_complete["year"] <= 2011)
        ]
        if not annual_complete.empty
        else annual_complete
    )
    if pre.empty and not annual_complete.empty:
        pre = annual_complete[annual_complete["year"] <= 2011]

    post = annual_complete[annual_complete["year"] >= 2013] if not annual_complete.empty else annual_complete
    if post.empty and not annual.empty:
        post = annual[annual["year"] >= 2013]

    cal_all_fallback = (
        bool(calibration_df["used_fallback"].all()) if not calibration_df.empty else True
    )
    cal_valid_ci = calibration_df[
        calibration_df["alpha_ci_low"].notna() & calibration_df["alpha_ci_high"].notna()
    ]
    alpha_ci_width_mean = (
        round(float((cal_valid_ci["alpha_ci_high"] - cal_valid_ci["alpha_ci_low"]).mean()), 4)
        if not cal_valid_ci.empty
        else np.nan
    )

    live_raw_points = float(obs_monthly["raw_total_points"].sum()) if not obs_monthly.empty else 0.0
    live_gridded_cells = float(obs_monthly["gridded_total_cells"].sum()) if not obs_monthly.empty else 0.0
    live_harmonized = float(obs_monthly["harmonized_hotspots"].sum()) if not obs_monthly.empty else 0.0

    is_short_window = pre.empty or post.empty or (obs_monthly["year"].nunique() < 2)
    if is_short_window and reference_metrics:
        return {
            "pre2012_start_year": int(reference_metrics.get("pre2012_start_year", pre2012_start_year)),
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
            "pre2012_modis_annual_mean": float(reference_metrics.get("pre2012_modis_annual_mean", 0.0)),
            "post2012_modis_annual_mean": float(reference_metrics.get("post2012_modis_annual_mean", 0.0)),
            "modis_natural_change_pct": float(reference_metrics.get("modis_natural_change_pct", 0.0)),
            "overlap_sensor_residual_bias_pct": float(
                reference_metrics.get("overlap_sensor_residual_bias_pct", 0.0)
            ),
            "holdout_bias_pct": float(reference_metrics.get("holdout_bias_pct", 0.0)),
            "deseasonalized_anomaly_r": float(
                reference_metrics.get("deseasonalized_anomaly_r", 0.0)
            ),
            "alpha_ci_width_mean": reference_metrics.get("alpha_ci_width_mean", alpha_ci_width_mean),
            "jump_reduction_pct": float(reference_metrics.get("jump_reduction_pct", 0.0)),
            "mean_alpha_day": float(reference_metrics.get("mean_alpha_day", 1.0)),
            "mean_alpha_night": float(reference_metrics.get("mean_alpha_night", 1.0)),
            "mean_alpha_overall": float(reference_metrics.get("mean_alpha_overall", 1.0)),
            "overlap_pearson_r": float(reference_metrics.get("overlap_pearson_r", 0.0)),
            "overlap_r2": float(reference_metrics.get("overlap_r2", 0.0)),
            "ols_slope": float(reference_metrics.get("ols_slope", 1.0)),
            "ols_intercept": float(reference_metrics.get("ols_intercept", 0.0)),
            "used_reference_calibration": True,
            "has_valid_overlap": True,
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

    # MODIS-only benchmark over the same pre/post periods to separate natural fire shift
    pre_modis_mean = float(pre["modis_active_cells"].mean()) if not pre.empty else 0.0
    post_modis_mean = float(post["modis_active_cells"].mean()) if not post.empty else 0.0

    def _pct_jump(after_val: float, before_val: float) -> float:
        if before_val <= 0:
            return 0.0
        return ((after_val - before_val) / before_val) * 100.0

    raw_combined_jump_pct = _pct_jump(post_raw_combined_mean, pre_raw_mean)
    raw_viirs_jump_pct = _pct_jump(post_raw_viirs_mean, pre_raw_mean)
    gridded_uncal_jump_pct = _pct_jump(post_gridded_viirs_mean, pre_harm_mean)
    harmonized_diff_pct = _pct_jump(post_harm_mean, pre_harm_mean)
    modis_natural_change_pct = _pct_jump(post_modis_mean, pre_modis_mean)

    # Concurrent overlap (2012-03 .. 2025-12) sensor residual bias: calibrated VIIRS vs MODIS
    ov_m = obs_monthly[
        (obs_monthly["date_label"] >= "2012-03")
        & (obs_monthly["year"] <= OVERLAP_YEARS[1])
        & (obs_monthly["year"] < current_year)
    ]
    if ov_m.empty and not obs_monthly.empty:
        ov_m = obs_monthly[
            (obs_monthly["date_label"] >= "2012-03")
            & (obs_monthly["year"] <= OVERLAP_YEARS[1])
        ]
    ov_modis = float(ov_m["modis_active_cells"].sum()) if not ov_m.empty else 0.0
    ov_viirs_cal = float(ov_m["viirs_calibrated_cells"].sum()) if not ov_m.empty else 0.0
    has_valid_overlap = bool(
        ov_modis > 0 and ov_viirs_cal > 0 and not cal_all_fallback
    )
    overlap_sensor_residual_bias_pct = (
        _pct_jump(ov_viirs_cal, ov_modis) if has_valid_overlap else 0.0
    )

    # Headline alpha_day / alpha_night / alpha_overall as exact overlap sum ratios
    if not has_valid_overlap:
        day_alpha = 1.0
        night_alpha = 1.0
        overall_alpha = 1.0
    elif gridded_df is not None and not gridded_df.empty:
        ov_g = gridded_df[
            (gridded_df["acq_date"] >= VIIRS_TRANSITION_DATE)
            & (gridded_df["year"] >= OVERLAP_YEARS[0])
            & (gridded_df["year"] <= OVERLAP_YEARS[1])
        ]
        m_tot = float(ov_g.loc[ov_g["sensor_group"] == "MODIS", "active_cell_count"].sum())
        v_tot = float(ov_g.loc[ov_g["sensor_group"] == "VIIRS", "active_cell_count"].sum())
        overall_alpha = (m_tot / v_tot) if (m_tot > 0 and v_tot > 0) else 1.0
        m_d = float(ov_g.loc[(ov_g["sensor_group"] == "MODIS") & (ov_g["daynight"] == "D"), "active_cell_count"].sum())
        v_d = float(ov_g.loc[(ov_g["sensor_group"] == "VIIRS") & (ov_g["daynight"] == "D"), "active_cell_count"].sum())
        day_alpha = (m_d / v_d) if (m_d > 0 and v_d > 0) else overall_alpha
        m_n = float(ov_g.loc[(ov_g["sensor_group"] == "MODIS") & (ov_g["daynight"] == "N"), "active_cell_count"].sum())
        v_n = float(ov_g.loc[(ov_g["sensor_group"] == "VIIRS") & (ov_g["daynight"] == "N"), "active_cell_count"].sum())
        night_alpha = (m_n / v_n) if (m_n > 0 and v_n > 0) else overall_alpha
    elif "overlap_alpha_overall" in corr_stats:
        day_alpha = float(corr_stats.get("overlap_alpha_day", 1.0))
        night_alpha = float(corr_stats.get("overlap_alpha_night", 1.0))
        overall_alpha = float(corr_stats.get("overlap_alpha_overall", 1.0))
    else:
        day_alpha = float(calibration_df.loc[calibration_df["daynight"] == "D", "alpha_cells"].mean())
        night_alpha = float(calibration_df.loc[calibration_df["daynight"] == "N", "alpha_cells"].mean())
        overall_alpha = float(calibration_df["alpha_cells"].mean())

    jump_reduction_pct = 0.0
    if has_valid_overlap and abs(raw_combined_jump_pct) > 1e-6:
        jump_reduction_pct = float(
            np.clip((1.0 - abs(harmonized_diff_pct) / abs(raw_combined_jump_pct)) * 100.0, 0.0, 100.0)
        )

    oos_metrics = (
        _compute_holdout_and_anomaly_metrics(monthly_df=obs_monthly, gridded_df=gridded_df)
        if has_valid_overlap
        else {"holdout_bias_pct": 0.0, "deseasonalized_anomaly_r": 0.0}
    )

    clean_corr = {
        k: v
        for k, v in corr_stats.items()
        if k in ("overlap_pearson_r", "overlap_r2", "ols_slope", "ols_intercept")
    }

    return {
        "pre2012_start_year": int(pre2012_start_year),
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
        "pre2012_modis_annual_mean": round(pre_modis_mean, 1),
        "post2012_modis_annual_mean": round(post_modis_mean, 1),
        "modis_natural_change_pct": round(modis_natural_change_pct, 2),
        "overlap_sensor_residual_bias_pct": round(overlap_sensor_residual_bias_pct, 2),
        "holdout_bias_pct": oos_metrics["holdout_bias_pct"],
        "deseasonalized_anomaly_r": oos_metrics["deseasonalized_anomaly_r"],
        "alpha_ci_width_mean": alpha_ci_width_mean,
        "jump_reduction_pct": round(jump_reduction_pct, 2),
        "mean_alpha_day": round(day_alpha, 4),
        "mean_alpha_night": round(night_alpha, 4),
        "mean_alpha_overall": round(overall_alpha, 4),
        "used_reference_calibration": False,
        "has_valid_overlap": has_valid_overlap,
        "live_raw_points": round(live_raw_points, 1),
        "live_gridded_cells": round(live_gridded_cells, 1),
        "live_harmonized_hotspots": round(live_harmonized, 1),
        **clean_corr,
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
        pre2012_start_year: int = 2003,
    ) -> None:
        self.modis_min_conf = modis_min_conf
        self.viirs_allowed_conf = list(
            viirs_allowed_conf if viirs_allowed_conf is not None else DEFAULT_VIIRS_CONFIDENCE
        )
        self.grid_step_deg = grid_step_deg
        self.overlap_years = overlap_years
        self.combination_mode = combination_mode
        self.daynight_filter = daynight_filter
        self.pre2012_start_year = int(pre2012_start_year)

    def run(
        self,
        modis_df: pd.DataFrame,
        viirs_df: pd.DataFrame,
        external_calibration_df: Optional[pd.DataFrame] = None,
        external_overlap_df: Optional[pd.DataFrame] = None,
        reference_transition_metrics: Optional[dict[str, float]] = None,
        calibration_override: Optional[pd.DataFrame] = None,
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
        if calibration_override is not None and not calibration_override.empty:
            calibration_df = calibration_override.copy()
        elif external_calibration_df is not None and not external_calibration_df.empty:
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
            gridded_df=gridded,
            pre2012_start_year=self.pre2012_start_year,
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
