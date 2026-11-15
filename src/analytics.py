"""Burning Activity Calendar analytics, historical climatology, anomaly scoring, and KPIs."""

from __future__ import annotations

import calendar
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, REGION_PRESETS
from src.data_loader import generate_regional_sample_data
from src.harmonizer import HarmonizationPipeline

MONTH_LABELS: list[str] = [calendar.month_abbr[m] for m in range(1, 13)]


def build_calendar_matrix(
    monthly_df: pd.DataFrame,
    value_col: str = "harmonized_hotspots",
) -> pd.DataFrame:
    """Pivot monthly series into a Year x Month matrix (columns 'Jan'..'Dec')."""
    if monthly_df.empty:
        return pd.DataFrame(columns=MONTH_LABELS)

    pivot = monthly_df.pivot_table(
        index="year",
        columns="month",
        values=value_col,
        aggfunc="sum",
        dropna=False,
    )
    for month in range(1, 13):
        if month not in pivot.columns:
            pivot[month] = np.nan
    pivot = pivot[list(range(1, 13))].sort_index()
    pivot.columns = MONTH_LABELS
    pivot.index.name = "year"
    return pivot


def compute_monthly_climatology(
    monthly_df: pd.DataFrame,
    value_col: str = "harmonized_hotspots",
    baseline_years: tuple[int, int] = (2001, 2025),
) -> pd.DataFrame:
    """Compute historical monthly median, mean, std, IQR (P25–P75), min, max, and peak year."""
    if monthly_df.empty:
        return pd.DataFrame(
            columns=[
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
                "peak_year",
            ]
        )

    b_start, b_end = baseline_years
    baseline = monthly_df[
        (monthly_df["year"] >= b_start) & (monthly_df["year"] <= b_end)
    ]
    if baseline.empty:
        baseline = monthly_df

    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        sub = baseline[baseline["month"] == month]
        if sub.empty:
            rows.append(
                {
                    "month": month,
                    "month_name": MONTH_LABELS[month - 1],
                    "median": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "p25": 0.0,
                    "p75": 0.0,
                    "iqr": 0.0,
                    "min": 0.0,
                    "max": 0.0,
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
                "peak_year": int(sub.loc[peak_idx, "year"]),
            }
        )
    return pd.DataFrame(rows)


def _classify_anomaly(row: pd.Series) -> str:
    """Assign human-readable anomaly status relative to historical distribution."""
    val = float(row["value"])
    if row["z_score"] > 1.5:
        return "Critical Anomaly (> +1.5σ)"
    if val > float(row["p75"]):
        return "Above Normal (> P75)"
    if val < float(row["p25"]):
        return "Below Normal (< P25)"
    return "Normal (P25–P75)"


def compute_monthly_anomalies(
    monthly_df: pd.DataFrame,
    climatology_df: pd.DataFrame,
    value_col: str = "harmonized_hotspots",
) -> pd.DataFrame:
    """Compute absolute, percentage, Z-score, and robust Z-score anomalies per month."""
    if monthly_df.empty:
        return pd.DataFrame()

    merged = monthly_df.merge(climatology_df, on="month", how="left").copy()
    merged["value"] = merged[value_col].astype(float)
    merged["anomaly_abs"] = np.round(merged["value"] - merged["median"], 2)
    merged["anomaly_pct"] = np.round(
        ((merged["value"] - merged["median"]) / np.maximum(merged["median"], 1.0)) * 100.0,
        1,
    )
    merged["z_score"] = np.round(
        (merged["value"] - merged["mean"]) / np.maximum(merged["std"], 1.0),
        2,
    )
    merged["robust_z_score"] = np.round(
        (merged["value"] - merged["median"]) / np.maximum(0.7413 * merged["iqr"], 1.0),
        2,
    )
    merged["anomaly_status"] = merged.apply(_classify_anomaly, axis=1)
    return merged


def compute_kpi_metrics(
    monthly_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    transition_metrics: dict[str, float],
    climatology_df: pd.DataFrame,
    target_year: Optional[int] = None,
    target_month: Optional[int] = None,
) -> dict[str, object]:
    """Compute dashboard KPI metrics for the selected region and reference period."""
    if monthly_df.empty or climatology_df.empty:
        return {}

    peak_row = climatology_df.loc[climatology_df["median"].idxmax()]

    annual = (
        monthly_df.groupby("year", as_index=False)
        .agg(
            annual_hotspots=("harmonized_hotspots", "sum"),
            annual_frp=("harmonized_frp", "sum"),
            annual_raw_points=("raw_total_points", "sum"),
        )
        .sort_values("year")
    )
    worst_row = annual.loc[annual["annual_hotspots"].idxmax()]

    ref_year = target_year if target_year in set(monthly_df["year"]) else int(monthly_df["year"].max())
    year_anom = anomalies_df[anomalies_df["year"] == ref_year]
    if target_month is not None and target_month in set(year_anom["month"]):
        focus_row = year_anom[year_anom["month"] == target_month].iloc[0]
    else:
        # Focus on the region's peak climatological month in the target year
        peak_m = int(peak_row["month"])
        sub_m = year_anom[year_anom["month"] == peak_m]
        focus_row = sub_m.iloc[0] if not sub_m.empty else year_anom.iloc[-1]

    return {
        "peak_month_num": int(peak_row["month"]),
        "peak_month_name": str(peak_row["month_name"]),
        "peak_month_median": float(peak_row["median"]),
        "worst_year": int(worst_row["year"]),
        "worst_year_hotspots": round(float(worst_row["annual_hotspots"]), 1),
        "worst_year_frp": round(float(worst_row["annual_frp"]), 1),
        "focus_year": int(focus_row["year"]),
        "focus_month_name": str(focus_row["month_name"]),
        "focus_value": round(float(focus_row["value"]), 1),
        "focus_median": round(float(focus_row["median"]), 1),
        "focus_anomaly_pct": float(focus_row["anomaly_pct"]),
        "focus_z_score": float(focus_row["z_score"]),
        "focus_status": str(focus_row["anomaly_status"]),
        "mean_alpha_overall": transition_metrics.get("mean_alpha_overall", 1.0),
        "mean_alpha_day": transition_metrics.get("mean_alpha_day", 1.0),
        "mean_alpha_night": transition_metrics.get("mean_alpha_night", 1.0),
        "jump_reduction_pct": transition_metrics.get("jump_reduction_pct", 0.0),
    }


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
        exported = export_processed_outputs(
            region_key,
            result.monthly_df,
            result.calibration_df,
            calendar_df,
        )

        tm = result.transition_metrics
        print(f"\n--- {preset['name']} ({region_key}) ---")
        print(
            f"  Raw Annual Mean (2001–2011 vs 2013–2026): "
            f"{tm['pre2012_raw_annual_mean']:,.1f} -> {tm['post2012_raw_combined_annual_mean']:,.1f} "
            f"({tm['raw_combined_jump_pct']:+.1f}% step jump)"
        )
        print(
            f"  Gridded Uncalibrated (2001–2011 vs VIIRS 2013–2026): "
            f"{tm['pre2012_gridded_annual_mean']:,.1f} -> {tm['post2012_gridded_viirs_annual_mean']:,.1f} "
            f"({tm['gridded_uncalibrated_jump_pct']:+.1f}%)"
        )
        print(
            f"  Harmonized Annual Mean (2001–2011 vs 2013–2026): "
            f"{tm['pre2012_gridded_annual_mean']:,.1f} -> {tm['post2012_harmonized_annual_mean']:,.1f} "
            f"({tm['harmonized_diff_pct']:+.1f}% natural diff)"
        )
        print(
            f"  Overlap Sensor Residual Bias (2012–2025 VIIRS_cal vs MODIS): "
            f"{tm['overlap_sensor_residual_bias_pct']:+.2f}% | "
            f"Out-of-Sample Holdout (2019–2025): {tm.get('holdout_bias_pct', 0.0):+.2f}%"
        )
        print(
            f"  2012 Step-Jump Elimination: {tm['jump_reduction_pct']:.2f}% | "
            f"Overlap R^2={tm['overlap_r2']:.4f} (r={tm['overlap_pearson_r']:.4f}, "
            f"Deseasonalized r={tm.get('deseasonalized_anomaly_r', 0.0):.4f})"
        )
        print(
            f"  Alpha VIIRS->MODIS: Overall={tm['mean_alpha_overall']:.4f} | "
            f"Day={tm['mean_alpha_day']:.4f} | Night={tm['mean_alpha_night']:.4f}"
        )
        print(
            f"  KPIs: Peak Month={kpis['peak_month_name']} (median={kpis['peak_month_median']}) | "
            f"Worst Year={kpis['worst_year']} ({kpis['worst_year_hotspots']:,.1f} cells) | "
            f"Focus {kpis['focus_year']}-{kpis['focus_month_name']}={kpis['focus_status']} ({kpis['focus_anomaly_pct']:+.1f}%)"
        )
        print(f"  Exported: {exported['monthly_csv']}")
