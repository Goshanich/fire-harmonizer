"""Burning Activity Calendar analytics, historical climatology, anomaly scoring, and KPIs."""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.config import PROCESSED_DIR, REAL_DIR, REGION_PRESETS
from src.data_loader import generate_regional_sample_data, load_real_cordoba_dataset
from src.harmonizer import HarmonizationPipeline, build_daily_sensor_breakdown

MONTH_LABELS: list[str] = [calendar.month_abbr[m] for m in range(1, 13)]


def build_calendar_matrix(
    monthly_df: pd.DataFrame,
    value_col: str = "harmonized_hotspots",
) -> pd.DataFrame:
    """Pivot monthly series into a Year x Month matrix (columns 'Jan'..'Dec'), preserving NaN for no_data."""
    if monthly_df.empty:
        return pd.DataFrame(columns=MONTH_LABELS)

    work = monthly_df.copy()
    if "data_status" in work.columns and value_col in work.columns:
        work.loc[work["data_status"] == "no_data", value_col] = np.nan

    pivot = work.pivot_table(
        index="year",
        columns="month",
        values=value_col,
        aggfunc=lambda s: np.nan if s.isna().all() else float(s.sum(skipna=True)),
        dropna=False,
    )
    for month in range(1, 13):
        if month not in pivot.columns:
            pivot[month] = np.nan
    pivot = pivot[list(range(1, 13))].sort_index()
    pivot.columns = MONTH_LABELS
    pivot.index.name = "year"
    return pivot


def _filter_observed_monthly(
    monthly_df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Filter monthly rows to genuinely observed non-NaN values (exclude 'no_data')."""
    if monthly_df.empty or value_col not in monthly_df.columns:
        return monthly_df.iloc[0:0].copy()
    work = monthly_df.copy()
    if "data_status" in work.columns:
        work = work[work["data_status"] != "no_data"]
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    return work[work[value_col].notna()].copy()


def compute_monthly_climatology(
    monthly_df: pd.DataFrame,
    value_col: str = "harmonized_hotspots",
    baseline_years: tuple[int, int] = (2001, 2025),
) -> pd.DataFrame:
    """Compute historical monthly median, mean, std, IQR (P25–P75), min, max, n_years, and peak year."""
    cols = [
        "month",
        "month_name",
        "median",
        "mean",
        "std",
        "p25",
        "p75",
        "iqr",
        "min",
        "max",
        "n_years",
        "peak_year",
    ]
    observed = _filter_observed_monthly(monthly_df, value_col)
    if observed.empty:
        return pd.DataFrame(columns=cols)

    b_start, b_end = baseline_years
    baseline = observed[
        (observed["year"] >= b_start) & (observed["year"] <= b_end)
    ]
    if baseline.empty:
        baseline = observed

    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        sub = baseline[baseline["month"] == month]
        if sub.empty:
            rows.append(
                {
                    "month": month,
                    "month_name": MONTH_LABELS[month - 1],
                    "median": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "iqr": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "n_years": 0,
                    "peak_year": None,
                }
            )
            continue

        vals = sub[value_col].to_numpy(dtype=float)
        p25, med, p75 = np.percentile(vals, [25, 50, 75])
        peak_idx = sub[value_col].idxmax()
        rows.append(
            {
                "month": month,
                "month_name": MONTH_LABELS[month - 1],
                "median": round(float(med), 2),
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 2),
                "p25": round(float(p25), 2),
                "p75": round(float(p75), 2),
                "iqr": round(float(p75 - p25), 2),
                "min": round(float(np.min(vals)), 2),
                "max": round(float(np.max(vals)), 2),
                "n_years": int(len(vals)),
                "peak_year": int(sub.loc[peak_idx, "year"]),
            }
        )
    return pd.DataFrame(rows)


def _classify_anomaly(row: pd.Series) -> str:
    """Assign human-readable anomaly status using Leave-One-Out Robust Z and IQR thresholds."""
    if str(row.get("data_status", "observed")) == "no_data" or pd.isna(row.get("value")):
        return "No Data"
    n_base = int(row.get("n_baseline_years", 0) or 0)
    if n_base < 8:
        return "Insufficient Data (<8 yrs)"

    val = float(row["value"])
    rz = float(row.get("robust_z_score", 0.0) or 0.0)
    p75 = float(row.get("p75", 0.0) or 0.0)
    p25 = float(row.get("p25", 0.0) or 0.0)
    if rz > 1.5 and val > p75:
        return "Critical Anomaly (Robust Z > +1.5)"
    if val > p75:
        return "Above Normal (> P75)"
    if val < p25:
        return "Below Normal (< P25)"
    return "Normal (P25–P75)"


def compute_monthly_anomalies(
    monthly_df: pd.DataFrame,
    climatology_df: pd.DataFrame,
    value_col: str = "harmonized_hotspots",
    baseline_years: tuple[int, int] = (2001, 2025),
) -> pd.DataFrame:
    """Compute Leave-One-Out anomalies, Z-scores, and Robust Z-scores (0.7413 * IQR) per month."""
    if monthly_df.empty:
        return pd.DataFrame()

    b_start, b_end = baseline_years
    observed = _filter_observed_monthly(monthly_df, value_col)
    in_base_window = observed[
        (observed["year"] >= b_start) & (observed["year"] <= b_end)
    ]
    base_pool = in_base_window if not in_base_window.empty else observed

    # Build lookup: month -> DataFrame(year, val) from observed baseline pool
    month_pools: dict[int, pd.DataFrame] = {}
    for m in range(1, 13):
        sub_m = base_pool[base_pool["month"] == m][["year", value_col]].copy()
        if sub_m.empty and not observed.empty:
            sub_m = observed[observed["month"] == m][["year", value_col]].copy()
        month_pools[m] = sub_m

    clim_meta = (
        climatology_df[["month", "month_name", "min", "max", "peak_year"]].copy()
        if not climatology_df.empty and "month" in climatology_df.columns
        else pd.DataFrame({"month": list(range(1, 13)), "month_name": MONTH_LABELS})
    )

    work = monthly_df.merge(clim_meta, on="month", how="left").copy()
    work["value"] = pd.to_numeric(work[value_col], errors="coerce")
    if "data_status" not in work.columns:
        work["data_status"] = np.where(work["value"].notna(), "observed", "no_data")

    meds, means, stds, p25s, p75s, iqrs, n_bases = [], [], [], [], [], [], []
    for _, row in work.iterrows():
        m = int(row["month"])
        y = int(row["year"])
        status = str(row.get("data_status", "observed"))
        val = row["value"]
        pool_m = month_pools.get(m, pd.DataFrame(columns=["year", value_col]))

        if status == "no_data" or pd.isna(val):
            loo_vals = pool_m[value_col].to_numpy(dtype=float) if not pool_m.empty else np.array([], dtype=float)
        else:
            loo_sub = pool_m[pool_m["year"] != y]
            loo_vals = loo_sub[value_col].to_numpy(dtype=float) if not loo_sub.empty else np.array([], dtype=float)

        n_bases.append(int(len(loo_vals)))
        if len(loo_vals) == 0:
            meds.append(np.nan)
            means.append(np.nan)
            stds.append(np.nan)
            p25s.append(np.nan)
            p75s.append(np.nan)
            iqrs.append(np.nan)
        else:
            p25, med, p75 = np.percentile(loo_vals, [25, 50, 75])
            meds.append(round(float(med), 2))
            means.append(round(float(np.mean(loo_vals)), 2))
            stds.append(round(float(np.std(loo_vals, ddof=1)) if len(loo_vals) > 1 else 0.0, 2))
            p25s.append(round(float(p25), 2))
            p75s.append(round(float(p75), 2))
            iqrs.append(round(float(p75 - p25), 2))

    work["median"] = meds
    work["mean"] = means
    work["std"] = stds
    work["p25"] = p25s
    work["p75"] = p75s
    work["iqr"] = iqrs
    work["n_baseline_years"] = n_bases

    is_obs = (work["data_status"] != "no_data") & work["value"].notna() & work["median"].notna()
    work["anomaly_abs"] = np.where(is_obs, np.round(work["value"] - work["median"], 2), np.nan)
    work["anomaly_pct"] = np.where(
        is_obs,
        np.round(((work["value"] - work["median"]) / np.maximum(work["median"], 1.0)) * 100.0, 1),
        np.nan,
    )
    work["z_score"] = np.where(
        is_obs & work["mean"].notna(),
        np.round((work["value"] - work["mean"]) / np.maximum(work["std"].fillna(0.0), 1.0), 2),
        np.nan,
    )
    work["robust_z_score"] = np.where(
        is_obs,
        np.round((work["value"] - work["median"]) / np.maximum(0.7413 * work["iqr"].fillna(0.0), 1.0), 2),
        np.nan,
    )
    work["anomaly_status"] = work.apply(_classify_anomaly, axis=1)
    return work


def compute_kpi_metrics(
    monthly_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    transition_metrics: dict[str, float],
    climatology_df: pd.DataFrame,
    target_year: Optional[int] = None,
    target_month: Optional[int] = None,
) -> dict[str, object]:
    """Compute dashboard KPI metrics for observed periods in the selected region."""
    observed = _filter_observed_monthly(monthly_df, "harmonized_hotspots")
    valid_clim = climatology_df.dropna(subset=["median"]) if not climatology_df.empty else climatology_df
    if observed.empty or valid_clim.empty:
        return {}

    peak_row = valid_clim.loc[valid_clim["median"].idxmax()]

    annual = (
        observed.groupby("year", as_index=False)
        .agg(
            annual_hotspots=("harmonized_hotspots", lambda s: float(s.sum(min_count=1)) if s.notna().any() else np.nan),
            annual_frp=("harmonized_frp", lambda s: float(s.sum(min_count=1)) if s.notna().any() else np.nan),
            annual_raw_points=("raw_total_points", "sum"),
        )
        .dropna(subset=["annual_hotspots"])
        .sort_values("year")
    )
    if annual.empty:
        return {}
    worst_row = annual.loc[annual["annual_hotspots"].idxmax()]

    obs_years = set(observed["year"].astype(int))
    ref_year = target_year if target_year in obs_years else int(max(obs_years))
    year_anom = anomalies_df[anomalies_df["year"] == ref_year]
    obs_year_anom = year_anom[year_anom["anomaly_status"] != "No Data"] if not year_anom.empty else year_anom

    if target_month is not None and target_month in set(obs_year_anom["month"]):
        focus_row = obs_year_anom[obs_year_anom["month"] == target_month].iloc[0]
    else:
        peak_m = int(peak_row["month"])
        sub_m = obs_year_anom[obs_year_anom["month"] == peak_m]
        if not sub_m.empty:
            focus_row = sub_m.iloc[0]
        elif not obs_year_anom.empty:
            focus_row = obs_year_anom.iloc[-1]
        else:
            focus_row = year_anom.iloc[0]

    focus_val = float(focus_row["value"]) if pd.notna(focus_row.get("value")) else 0.0
    focus_med = float(focus_row["median"]) if pd.notna(focus_row.get("median")) else 0.0
    focus_pct = float(focus_row["anomaly_pct"]) if pd.notna(focus_row.get("anomaly_pct")) else 0.0
    focus_z = float(focus_row["z_score"]) if pd.notna(focus_row.get("z_score")) else 0.0
    focus_rz = float(focus_row["robust_z_score"]) if pd.notna(focus_row.get("robust_z_score")) else 0.0

    return {
        "peak_month_num": int(peak_row["month"]),
        "peak_month_name": str(peak_row["month_name"]),
        "peak_month_median": float(peak_row["median"]),
        "worst_year": int(worst_row["year"]),
        "worst_year_hotspots": round(float(worst_row["annual_hotspots"]), 1),
        "worst_year_frp": round(float(worst_row["annual_frp"]), 1) if pd.notna(worst_row["annual_frp"]) else 0.0,
        "focus_year": int(focus_row["year"]),
        "focus_month_name": str(focus_row["month_name"]),
        "focus_value": round(focus_val, 1),
        "focus_median": round(focus_med, 1),
        "focus_anomaly_pct": focus_pct,
        "focus_z_score": focus_z,
        "focus_robust_z_score": focus_rz,
        "focus_status": str(focus_row["anomaly_status"]),
        "mean_alpha_overall": transition_metrics.get("mean_alpha_overall", 1.0),
        "mean_alpha_day": transition_metrics.get("mean_alpha_day", 1.0),
        "mean_alpha_night": transition_metrics.get("mean_alpha_night", 1.0),
        "jump_reduction_pct": transition_metrics.get("jump_reduction_pct", 0.0),
    }


def check_series_nonstationarity(
    monthly_df: pd.DataFrame,
    value_col: str = "harmonized_hotspots",
) -> dict[str, object]:
    """Detect statistically significant multi-year fire regime shift across complete observed years."""
    observed = _filter_observed_monthly(monthly_df, value_col)
    if observed.empty:
        return {
            "has_significant_trend": False,
            "n_years": 0,
            "slope_pct_per_decade": 0.0,
            "p_value": 1.0,
            "half_split_diff_pct": 0.0,
            "message": "",
        }

    annual = (
        observed.groupby("year", as_index=False)
        .agg(
            annual_val=(value_col, "sum"),
            obs_months=("month", "nunique"),
        )
        .sort_values("year")
    )
    annual = annual[annual["obs_months"] >= 6].reset_index(drop=True)
    if len(annual) < 6:
        return {
            "has_significant_trend": False,
            "n_years": int(len(annual)),
            "slope_pct_per_decade": 0.0,
            "p_value": 1.0,
            "half_split_diff_pct": 0.0,
            "message": "",
        }

    years = annual["year"].to_numpy(dtype=float)
    vals = annual["annual_val"].to_numpy(dtype=float)
    mean_val = float(np.mean(vals))
    if np.std(vals) < 1e-9 or mean_val <= 0:
        slope, p_val = 0.0, 1.0
    else:
        reg = linregress(years, vals)
        slope = float(reg.slope)
        p_val = float(reg.pvalue) if np.isfinite(reg.pvalue) else 1.0

    slope_pct_decade = round((slope * 10.0 / max(mean_val, 1e-6)) * 100.0, 1)
    mid = len(vals) // 2
    first_mean = float(np.mean(vals[:mid]))
    second_mean = float(np.mean(vals[mid:]))
    half_diff_pct = round(((second_mean - first_mean) / max(first_mean, 1e-6)) * 100.0, 1)

    has_trend = bool(p_val < 0.05 and abs(half_diff_pct) >= 15.0)
    direction = "decline" if slope < 0 else "increase"
    msg = (
        f"Non-stationary baseline detected across {len(annual)} observed years: "
        f"harmonized activity shows a statistically significant {direction} "
        f"({slope_pct_decade:+.1f}%/decade, half-period shift {half_diff_pct:+.1f}%, p = {p_val:.4f}). "
        "Long-term climatological anomalies reflect this multi-year regime shift."
        if has_trend
        else ""
    )
    return {
        "has_significant_trend": has_trend,
        "n_years": int(len(annual)),
        "slope_pct_per_decade": slope_pct_decade,
        "p_value": round(p_val, 4),
        "half_split_diff_pct": half_diff_pct,
        "message": msg,
    }


def compute_critical_fire_windows(
    daily_df: pd.DataFrame,
    value_col: str = "harmonized_cells",
    threshold_percentile: float = 75.0,
) -> pd.DataFrame:
    """Compute 52-week exceedance probability P(weekly activity > P75) to pinpoint critical fire windows."""
    cols = [
        "week",
        "week_label",
        "start_date_label",
        "p_exceedance",
        "threshold_p75",
        "mean_weekly",
        "median_weekly",
        "max_weekly",
        "n_years",
        "is_critical",
    ]
    if daily_df is None or daily_df.empty or "acq_date" not in daily_df.columns:
        return pd.DataFrame(columns=cols)

    if value_col in daily_df.columns:
        col = value_col
    elif "harmonized_cells" in daily_df.columns:
        col = "harmonized_cells"
    elif "calibrated_cell_count" in daily_df.columns:
        col = "calibrated_cell_count"
    else:
        return pd.DataFrame(columns=cols)

    work = daily_df[["acq_date", col]].copy()
    work["acq_dt"] = pd.to_datetime(work["acq_date"], errors="coerce")
    work = work.dropna(subset=["acq_dt"])
    if work.empty:
        return pd.DataFrame(columns=cols)

    daily_sums = work.groupby("acq_dt", as_index=False)[col].sum()
    full_days = pd.DataFrame({"acq_dt": pd.date_range(work["acq_dt"].min(), work["acq_dt"].max(), freq="D")})
    full_days = full_days.merge(daily_sums, on="acq_dt", how="left")
    full_days[col] = full_days[col].fillna(0.0)
    full_days["year"] = full_days["acq_dt"].dt.year
    full_days["week"] = ((full_days["acq_dt"].dt.day_of_year - 1) // 7).clip(upper=51) + 1

    weekly = (
        full_days.groupby(["year", "week"], as_index=False)
        .agg(weekly_val=(col, "sum"), obs_days=("acq_dt", "count"))
    )
    weekly = weekly[weekly["obs_days"] >= 4].copy()
    if weekly.empty:
        return pd.DataFrame(columns=cols)

    all_vals = weekly["weekly_val"].to_numpy(dtype=float)
    threshold = float(np.percentile(all_vals, threshold_percentile))
    if threshold <= 0.0 and np.any(all_vals > 0.0):
        pos_vals = all_vals[all_vals > 0.0]
        threshold = float(np.percentile(pos_vals, threshold_percentile))

    ref_jan1 = datetime(2025, 1, 1)
    rows: list[dict[str, object]] = []
    for w in range(1, 53):
        sub_w = weekly[weekly["week"] == w]["weekly_val"].to_numpy(dtype=float)
        start_dt = ref_jan1 + timedelta(days=(w - 1) * 7)
        date_lbl = start_dt.strftime("%b %d")
        w_label = f"W{w:02d} ({date_lbl})"
        if len(sub_w) == 0:
            rows.append(
                {
                    "week": w,
                    "week_label": w_label,
                    "start_date_label": date_lbl,
                    "p_exceedance": 0.0,
                    "threshold_p75": round(threshold, 2),
                    "mean_weekly": 0.0,
                    "median_weekly": 0.0,
                    "max_weekly": 0.0,
                    "n_years": 0,
                    "is_critical": False,
                }
            )
            continue

        if threshold > 0.0:
            p_exc = float(np.mean(sub_w > threshold))
        else:
            p_exc = float(np.mean(sub_w > 0.0))

        rows.append(
            {
                "week": w,
                "week_label": w_label,
                "start_date_label": date_lbl,
                "p_exceedance": round(p_exc, 3),
                "threshold_p75": round(threshold, 2),
                "mean_weekly": round(float(np.mean(sub_w)), 2),
                "median_weekly": round(float(np.median(sub_w)), 2),
                "max_weekly": round(float(np.max(sub_w)), 2),
                "n_years": int(len(sub_w)),
                "is_critical": bool(p_exc >= 0.50),
            }
        )
    return pd.DataFrame(rows)


def export_processed_outputs(
    region_key: str,
    monthly_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> dict[str, str]:
    """Persist processed harmonization tables and calendar matrices on E:\\."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    monthly_path = PROCESSED_DIR / f"harmonized_monthly_{region_key}.csv"
    cal_path = PROCESSED_DIR / f"calibration_alpha_{region_key}.csv"
    calendar_path = PROCESSED_DIR / f"burning_calendar_{region_key}.csv"

    monthly_df.to_csv(monthly_path, index=False)
    calibration_df.to_csv(cal_path, index=False)
    calendar_df.to_csv(calendar_path)

    return {
        "monthly_csv": str(monthly_path),
        "calibration_csv": str(cal_path),
        "calendar_csv": str(calendar_path),
    }


if __name__ == "__main__":
    print("=== Step 3 Verification: 4-Stage Harmonization & Burning Activity Calendar ===")
    pipeline = HarmonizationPipeline()

    for region_key, preset in REGION_PRESETS.items():
        modis_df, viirs_df = generate_regional_sample_data(region_key=region_key, force=False)
        result = pipeline.run(modis_df, viirs_df)

        calendar_df = build_calendar_matrix(result.monthly_df, value_col="harmonized_hotspots")
        climatology_df = compute_monthly_climatology(result.monthly_df)
        anomalies_df = compute_monthly_anomalies(result.monthly_df, climatology_df)
        kpis = compute_kpi_metrics(
            result.monthly_df,
            anomalies_df,
            result.transition_metrics,
            climatology_df,
        )
        nonstat = check_series_nonstationarity(result.monthly_df)
        daily_df = build_daily_sensor_breakdown(result.gridded_df, result.harmonized_cells_df)
        crit_win = compute_critical_fire_windows(daily_df)
        top_win = (
            crit_win.sort_values(["p_exceedance", "mean_weekly"], ascending=False).iloc[0]
            if not crit_win.empty
            else None
        )
        exported = export_processed_outputs(
            region_key,
            result.monthly_df,
            result.calibration_df,
            calendar_df,
        )

        tm = result.transition_metrics
        pre_s = int(tm.get("pre2012_start_year", 2003))
        print(f"\n--- {preset['name']} ({region_key}) ---")
        print(
            f"  Raw Annual Mean ({pre_s}–2011 vs 2013–2025): "
            f"{tm['pre2012_raw_annual_mean']:,.1f} -> {tm['post2012_raw_combined_annual_mean']:,.1f} "
            f"({tm['raw_combined_jump_pct']:+.1f}% step jump)"
        )
        print(
            f"  Gridded Uncalibrated ({pre_s}–2011 vs VIIRS 2013–2025): "
            f"{tm['pre2012_gridded_annual_mean']:,.1f} -> {tm['post2012_gridded_viirs_annual_mean']:,.1f} "
            f"({tm['gridded_uncalibrated_jump_pct']:+.1f}%)"
        )
        print(
            f"  Harmonized Annual Mean ({pre_s}–2011 vs 2013–2025): "
            f"{tm['pre2012_gridded_annual_mean']:,.1f} -> {tm['post2012_harmonized_annual_mean']:,.1f} "
            f"({tm['harmonized_diff_pct']:+.2f}% harmonized diff vs MODIS natural {tm.get('modis_natural_change_pct', 0.0):+.2f}%)"
        )
        print(
            f"  Overlap Sensor Residual Bias (2012-03..2025 VIIRS_cal vs MODIS): "
            f"{tm['overlap_sensor_residual_bias_pct']:+.2f}% | "
            f"Out-of-Sample Holdout (2019–2025): {tm.get('holdout_bias_pct', 0.0):+.2f}%"
        )
        print(
            f"  Step-Jump Suppression: {tm['jump_reduction_pct']:.2f}% | "
            f"Overlap R^2={tm['overlap_r2']:.4f} (r={tm['overlap_pearson_r']:.4f}, "
            f"Deseasonalized r={tm.get('deseasonalized_anomaly_r', 0.0):.4f})"
        )
        print(
            f"  Sum-Ratio Alpha VIIRS->MODIS: Overall={tm['mean_alpha_overall']:.4f} | "
            f"Day={tm['mean_alpha_day']:.4f} | Night={tm['mean_alpha_night']:.4f} | "
            f"Mean 95% CI Width={tm.get('alpha_ci_width_mean', 0.0):.4f}"
        )
        print(
            f"  KPIs: Peak Month={kpis['peak_month_name']} (median={kpis['peak_month_median']}) | "
            f"Worst Year={kpis['worst_year']} ({kpis['worst_year_hotspots']:,.1f} cells) | "
            f"Focus {kpis['focus_year']}-{kpis['focus_month_name']}={kpis['focus_status']} ({kpis['focus_anomaly_pct']:+.1f}%)"
        )
        if top_win is not None:
            print(
                f"  Peak Critical Window: {top_win['week_label']} "
                f"(P>P75={top_win['p_exceedance']*100:.1f}%, threshold={top_win['threshold_p75']:.1f}) | "
                f"Non-stationary={nonstat['has_significant_trend']} (p={nonstat['p_value']})"
            )
        print(f"  Exported: {exported['monthly_csv']}")

    real_modis_path = REAL_DIR / "modis_cordoba_real_2001_2025.csv"
    real_viirs_path = REAL_DIR / "viirs_cordoba_real_2012_2025.csv"
    if real_modis_path.exists() and real_viirs_path.exists():
        real_modis, real_viirs = load_real_cordoba_dataset()
        real_res = pipeline.run(real_modis, real_viirs)
        real_cal = build_calendar_matrix(real_res.monthly_df, value_col="harmonized_hotspots")
        real_clim = compute_monthly_climatology(real_res.monthly_df)
        real_anom = compute_monthly_anomalies(real_res.monthly_df, real_clim)
        real_kpis = compute_kpi_metrics(
            real_res.monthly_df,
            real_anom,
            real_res.transition_metrics,
            real_clim,
        )
        real_nonstat = check_series_nonstationarity(real_res.monthly_df)
        real_daily = build_daily_sensor_breakdown(real_res.gridded_df, real_res.harmonized_cells_df)
        real_crit = compute_critical_fire_windows(real_daily)
        real_top_win = (
            real_crit.sort_values(["p_exceedance", "mean_weekly"], ascending=False).iloc[0]
            if not real_crit.empty
            else None
        )
        real_exported = export_processed_outputs(
            "cordoba_real",
            real_res.monthly_df,
            real_res.calibration_df,
            real_cal,
        )
        rtm = real_res.transition_metrics
        pre_s = int(rtm.get("pre2012_start_year", 2003))
        print("\n--- Real NASA FIRMS Archive: Córdoba Province (2001–2025) ---")
        print(
            f"  Rows: MODIS={len(real_modis):,}, VIIRS={len(real_viirs):,} | "
            f"Sum-Ratio Alpha Overall={rtm['mean_alpha_overall']:.4f} "
            f"(D={rtm['mean_alpha_day']:.4f}, N={rtm['mean_alpha_night']:.4f}, "
            f"CI Width={rtm.get('alpha_ci_width_mean', 0.0):.4f})"
        )
        print(
            f"  Raw ({pre_s}–2011 vs 2013–2025): {rtm['pre2012_raw_annual_mean']:,.1f} -> "
            f"{rtm['post2012_raw_combined_annual_mean']:,.1f} ({rtm['raw_combined_jump_pct']:+.2f}%) | "
            f"Harmonized: {rtm['pre2012_gridded_annual_mean']:,.1f} -> {rtm['post2012_harmonized_annual_mean']:,.1f} "
            f"({rtm['harmonized_diff_pct']:+.2f}% vs MODIS natural {rtm['modis_natural_change_pct']:+.2f}%)"
        )
        print(
            f"  Jump Suppression={rtm['jump_reduction_pct']:.2f}% | "
            f"Overlap Bias={rtm['overlap_sensor_residual_bias_pct']:+.2f}% | "
            f"Holdout Bias={rtm['holdout_bias_pct']:+.2f}% | "
            f"R^2={rtm['overlap_r2']:.4f} | Deseasonalized r={rtm['deseasonalized_anomaly_r']:.4f}"
        )
        print(
            f"  KPIs: Peak={real_kpis['peak_month_name']} (median={real_kpis['peak_month_median']}) | "
            f"Worst Year={real_kpis['worst_year']} ({real_kpis['worst_year_hotspots']:,.1f}) | "
            f"Exported: {real_exported['calibration_csv']}"
        )
        if real_top_win is not None:
            print(
                f"  Peak Critical Window: {real_top_win['week_label']} "
                f"(P>P75={real_top_win['p_exceedance']*100:.1f}%, threshold={real_top_win['threshold_p75']:.1f}) | "
                f"Non-stationary={real_nonstat['has_significant_trend']} "
                f"(slope={real_nonstat['slope_pct_per_decade']:+.1f}%/decade, p={real_nonstat['p_value']})"
            )
