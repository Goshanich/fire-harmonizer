"""Interactive Streamlit dashboard for MODIS & VIIRS Hot Spots Harmonization.

NASA Space Apps Challenge 2026 — Harmonization of MODIS and VIIRS Hot Spots (NASA & CONAE).
"""

from __future__ import annotations

from datetime import date, timedelta
import gc
import json
from typing import Any, Optional

import folium
from folium.plugins import HeatMap
from folium.template import Template
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyproj import Geod
import streamlit as st

try:
    from shapely.geometry import shape as _shapely_shape
except ImportError:  # pragma: no cover - optional geometry helper
    _shapely_shape = None

from src.analytics import (
    MONTH_LABELS,
    build_calendar_matrix,
    check_series_nonstationarity,
    compute_critical_fire_windows,
    compute_kpi_metrics,
    compute_monthly_anomalies,
    compute_monthly_climatology,
)
from src.config import (
    COLOR_PALETTES,
    COLOR_SCALE_MODES,
    CONAE_CATALOG_OGC_URL,
    CONAE_FOCOS_URL,
    CONAE_WMS_BASE_URL,
    DEFAULT_MODIS_MIN_CONFIDENCE,
    DEFAULT_VIIRS_CONFIDENCE,
    FIRMS_ARCHIVE_URL,
    FIRMS_MAP_KEY_URL,
    FIRMS_MAP_URL,
    FIRMS_PORTAL_URL,
    PROCESSED_DIR,
    QUICK_LOCATIONS,
    REGION_PRESETS,
    YEAR_RANGE,
    build_firms_map_url,
    validate_bbox,
)
from src.data_loader import (
    FIRMSApiClient,
    FIRMSApiError,
    filter_by_bbox_and_dates,
    generate_regional_sample_data,
    load_csv_dataset,
    load_csv_dataset_with_report,
    load_real_cordoba_dataset,
)
from src.harmonizer import (
    HarmonizationPipeline,
    HarmonizationResult,
    build_daily_sensor_breakdown,
    build_in_window_calibration,
    grid_spatiotemporal_clusters,
    normalize_pixel_geometry,
    unify_and_filter_sensors,
)

st.set_page_config(
    page_title="MODIS & VIIRS Fire Harmonizer | NASA Space Apps 2026",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

_GEOD = Geod(ellps="WGS84")
PLOTLY_CONFIG: dict[str, Any] = {
    "scrollZoom": False,
    "displayModeBar": False,
    "doubleClick": False,
}


# ---------------------------------------------------------------------------
# Cached Data & Pipeline Helpers
# ---------------------------------------------------------------------------
@st.cache_data(
    max_entries=2,
    ttl=1800,
    show_spinner="Computing MODIS & VIIRS harmonization...",
)
def _load_sample_cached(region_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_regional_sample_data(region_key=region_key, force=False)


@st.cache_data(
    max_entries=2,
    ttl=1800,
    show_spinner="Loading real 2001–2025 NASA FIRMS archive...",
)
def _load_real_cordoba_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_real_cordoba_dataset()


def _execute_harmonization(
    modis_df: pd.DataFrame,
    viirs_df: pd.DataFrame,
    modis_min_conf: int,
    viirs_allowed_conf: tuple[str, ...],
    combination_mode: str,
    daynight_filter: str,
    external_calibration_df: Optional[pd.DataFrame] = None,
    external_overlap_df: Optional[pd.DataFrame] = None,
    reference_transition_metrics: Optional[dict[str, float]] = None,
    calibration_override: Optional[pd.DataFrame] = None,
) -> HarmonizationResult:
    """Run HarmonizationPipeline and drop unused raw filtered points to save RAM."""
    pipeline = HarmonizationPipeline(
        modis_min_conf=modis_min_conf,
        viirs_allowed_conf=list(viirs_allowed_conf),
        combination_mode=combination_mode,
        daynight_filter=daynight_filter,
    )
    res = pipeline.run(
        modis_df,
        viirs_df,
        external_calibration_df=external_calibration_df,
        external_overlap_df=external_overlap_df,
        reference_transition_metrics=reference_transition_metrics,
        calibration_override=calibration_override,
    )
    res.filtered_points_df = pd.DataFrame()
    return res


@st.cache_data(
    max_entries=2,
    ttl=1800,
    show_spinner=False,
)
def _compute_in_window_calibration_cached(
    modis_df: pd.DataFrame,
    viirs_df: pd.DataFrame,
    modis_min_conf: int,
    viirs_allowed_conf: tuple[str, ...],
    daynight_filter: str,
    bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    source_key: str = "",
    start_date: str = "",
    end_date: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _ = (bbox, source_key, start_date, end_date)
    unified = unify_and_filter_sensors(
        modis_df=modis_df,
        viirs_df=viirs_df,
        modis_min_conf=modis_min_conf,
        viirs_allowed_conf=list(viirs_allowed_conf),
        daynight_filter=daynight_filter,
    )
    normalized = normalize_pixel_geometry(unified)
    gridded = grid_spatiotemporal_clusters(normalized)
    return build_in_window_calibration(gridded)


@st.cache_data(
    max_entries=2,
    ttl=1800,
    show_spinner="Computing MODIS & VIIRS harmonization...",
)
def _run_harmonization_cached(
    modis_df: pd.DataFrame,
    viirs_df: pd.DataFrame,
    modis_min_conf: int,
    viirs_allowed_conf: tuple[str, ...],
    combination_mode: str,
    daynight_filter: str,
    external_calibration_df: Optional[pd.DataFrame] = None,
    external_overlap_df: Optional[pd.DataFrame] = None,
    reference_transition_metrics: Optional[dict[str, float]] = None,
    calibration_override: Optional[pd.DataFrame] = None,
    bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    source_key: str = "",
    start_date: str = "",
    end_date: str = "",
) -> HarmonizationResult:
    _ = (bbox, source_key, start_date, end_date)
    return _execute_harmonization(
        modis_df=modis_df,
        viirs_df=viirs_df,
        modis_min_conf=modis_min_conf,
        viirs_allowed_conf=viirs_allowed_conf,
        combination_mode=combination_mode,
        daynight_filter=daynight_filter,
        external_calibration_df=external_calibration_df,
        external_overlap_df=external_overlap_df,
        reference_transition_metrics=reference_transition_metrics,
        calibration_override=calibration_override,
    )


@st.cache_data(max_entries=2, ttl=1800, show_spinner=False)
def _load_regional_calibration_csv(region_key: str) -> Optional[pd.DataFrame]:
    cal_path = PROCESSED_DIR / f"calibration_alpha_{region_key}.csv"
    if cal_path.exists():
        try:
            df = pd.read_csv(cal_path)
            if {"month", "daynight", "alpha_cells", "alpha_frp"}.issubset(df.columns):
                return df
        except Exception:
            return None
    return None


@st.cache_data(
    max_entries=2,
    ttl=1800,
    show_spinner="Computing MODIS & VIIRS harmonization...",
)
def _load_regional_reference_cached(
    region_key: str,
    modis_min_conf: int,
    viirs_allowed_conf: tuple[str, ...],
    combination_mode: str,
    daynight_filter: str,
    source_kind: str = "synthetic",
) -> tuple[HarmonizationResult, pd.DataFrame]:
    """Load pre-computed calibration & multi-decade reference baseline for short windows."""
    if source_kind == "real_cordoba":
        ref_modis, ref_viirs = _load_real_cordoba_cached()
        cal_key = "cordoba_real"
    else:
        ref_modis, ref_viirs = _load_sample_cached(region_key)
        cal_key = region_key
    ref_modis_dn = filter_by_bbox_and_dates(ref_modis, daynight=daynight_filter)
    ref_viirs_dn = filter_by_bbox_and_dates(ref_viirs, daynight=daynight_filter)

    ref_result = _execute_harmonization(
        modis_df=ref_modis_dn,
        viirs_df=ref_viirs_dn,
        modis_min_conf=modis_min_conf,
        viirs_allowed_conf=viirs_allowed_conf,
        combination_mode=combination_mode,
        daynight_filter=daynight_filter,
    )

    use_default_csv = (
        daynight_filter == "All"
        and modis_min_conf == DEFAULT_MODIS_MIN_CONFIDENCE
        and tuple(sorted(viirs_allowed_conf)) == tuple(sorted(DEFAULT_VIIRS_CONFIDENCE))
    )
    cal_df = _load_regional_calibration_csv(cal_key) if use_default_csv else None
    if cal_df is None or cal_df.empty:
        cal_df = ref_result.calibration_df.copy()

    return ref_result, cal_df


def _compute_climatology_bundle(
    monthly_df: pd.DataFrame,
    gridded_df: pd.DataFrame,
    transition_metrics: dict[str, Any],
    ref_monthly_df: Optional[pd.DataFrame],
    metric_choice: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute calendar matrix, seasonal climatology, anomalies, and KPIs."""
    calendar_df = build_calendar_matrix(monthly_df, value_col=metric_choice)
    is_ref_mode = bool(transition_metrics.get("used_reference_calibration", False))
    if is_ref_mode and ref_monthly_df is not None:
        climatology_df = compute_monthly_climatology(ref_monthly_df, value_col=metric_choice)
        active_live_month = (
            int(gridded_df["month"].mode().iloc[0])
            if not gridded_df.empty
            else None
        )
    else:
        climatology_df = compute_monthly_climatology(monthly_df, value_col=metric_choice)
        active_live_month = None

    anomalies_df = compute_monthly_anomalies(monthly_df, climatology_df, value_col=metric_choice)
    kpis = compute_kpi_metrics(
        monthly_df=monthly_df,
        anomalies_df=anomalies_df,
        transition_metrics=transition_metrics,
        climatology_df=climatology_df,
        target_year=int(monthly_df["year"].max()),
        target_month=active_live_month,
    )
    return calendar_df, climatology_df, anomalies_df, kpis


@st.cache_data(
    max_entries=2,
    ttl=1800,
    show_spinner="Computing MODIS & VIIRS harmonization...",
)
def _compute_climatology_bundle_cached(
    monthly_df: pd.DataFrame,
    gridded_df: pd.DataFrame,
    transition_metrics: dict[str, Any],
    ref_monthly_df: Optional[pd.DataFrame],
    metric_choice: str,
    bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    source_key: str = "",
    start_date: str = "",
    end_date: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute and cache calendar matrix, seasonal climatology, anomalies, and KPIs."""
    _ = (bbox, source_key, start_date, end_date)
    return _compute_climatology_bundle(
        monthly_df=monthly_df,
        gridded_df=gridded_df,
        transition_metrics=transition_metrics,
        ref_monthly_df=ref_monthly_df,
        metric_choice=metric_choice,
    )


@st.cache_data(
    max_entries=2,
    ttl=1800,
    show_spinner="Computing MODIS & VIIRS harmonization...",
)
def _run_preset_full_bundle_cached(
    region_key: str,
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    daynight_filter: str,
    modis_min_conf: int,
    viirs_allowed_conf: tuple[str, ...],
    combination_mode: str,
    metric_choice: str,
    source_kind: str = "synthetic",
) -> tuple[
    int,
    int,
    HarmonizationResult,
    Optional[HarmonizationResult],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """End-to-end cached archive loading, filtering, harmonization, and climatology."""
    if source_kind == "real_cordoba":
        raw_modis_df, raw_viirs_df = _load_real_cordoba_cached()
    else:
        raw_modis_df, raw_viirs_df = _load_sample_cached(region_key)
    modis_filtered = filter_by_bbox_and_dates(
        raw_modis_df,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        daynight=daynight_filter,
    )
    viirs_filtered = filter_by_bbox_and_dates(
        raw_viirs_df,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        daynight=daynight_filter,
    )
    modis_count = len(modis_filtered)
    viirs_count = len(viirs_filtered)

    all_dates = pd.concat(
        [
            modis_filtered["acq_date"] if not modis_filtered.empty else pd.Series(dtype=str),
            viirs_filtered["acq_date"] if not viirs_filtered.empty else pd.Series(dtype=str),
        ],
        ignore_index=True,
    )
    unique_years_count = int(all_dates.astype(str).str[:4].nunique()) if not all_dates.empty else 0

    ref_result: Optional[HarmonizationResult] = None
    ext_cal_df: Optional[pd.DataFrame] = None
    ext_overlap_df: Optional[pd.DataFrame] = None
    ref_tm: Optional[dict[str, float]] = None
    if unique_years_count < 2:
        ref_result, ext_cal_df = _load_regional_reference_cached(
            region_key=region_key,
            modis_min_conf=modis_min_conf,
            viirs_allowed_conf=viirs_allowed_conf,
            combination_mode=combination_mode,
            daynight_filter=daynight_filter,
            source_kind=source_kind,
        )
        ext_overlap_df = ref_result.overlap_monthly_df
        ref_tm = ref_result.transition_metrics

    result = _execute_harmonization(
        modis_df=modis_filtered,
        viirs_df=viirs_filtered,
        modis_min_conf=modis_min_conf,
        viirs_allowed_conf=viirs_allowed_conf,
        combination_mode=combination_mode,
        daynight_filter=daynight_filter,
        external_calibration_df=ext_cal_df,
        external_overlap_df=ext_overlap_df,
        reference_transition_metrics=ref_tm,
    )
    result.filtered_points_df = pd.DataFrame()
    del modis_filtered, viirs_filtered, all_dates

    if result.monthly_df.empty:
        empty_df = pd.DataFrame()
        gc.collect()
        return modis_count, viirs_count, result, ref_result, empty_df, empty_df, empty_df, {}

    calendar_df, climatology_df, anomalies_df, kpis = _compute_climatology_bundle(
        monthly_df=result.monthly_df,
        gridded_df=result.gridded_df,
        transition_metrics=result.transition_metrics,
        ref_monthly_df=ref_result.monthly_df if ref_result is not None else None,
        metric_choice=metric_choice,
    )
    gc.collect()
    return (
        modis_count,
        viirs_count,
        result,
        ref_result,
        calendar_df,
        climatology_df,
        anomalies_df,
        kpis,
    )


def _get_sensor_date_bounds(
    avail_df: pd.DataFrame,
    sources: list[str],
) -> tuple[dict[str, tuple[date, date]], Optional[date], Optional[date]]:
    """Extract (min_date, max_date) bounds per FIRMS source and their overlap window."""
    per_source: dict[str, tuple[date, date]] = {}
    if avail_df is None or avail_df.empty or "data_id" not in avail_df.columns:
        return per_source, None, None

    for src in sources:
        sub = avail_df[avail_df["data_id"].astype(str) == src]
        if sub.empty:
            continue
        try:
            min_d = date.fromisoformat(str(sub.iloc[0]["min_date"])[:10])
            max_d = date.fromisoformat(str(sub.iloc[0]["max_date"])[:10])
            per_source[src] = (min_d, max_d)
        except Exception:
            continue

    if not per_source:
        return per_source, None, None

    common_min = max(bounds[0] for bounds in per_source.values())
    common_max = min(bounds[1] for bounds in per_source.values())
    return per_source, common_min, common_max


def _compute_bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    west, south, east, north = bbox
    if east <= west or north <= south:
        return 0.0
    lons = [west, east, east, west, west]
    lats = [south, south, north, north, south]
    area_m2, _ = _GEOD.polygon_area_perimeter(lons, lats)
    return abs(area_m2) / 1_000_000.0


def _extract_bbox_from_geojson(payload: Any) -> Optional[tuple[float, float, float, float]]:
    """Extract (west, south, east, north) bounds from a GeoJSON object."""
    if not isinstance(payload, dict):
        return None
    if _shapely_shape is not None:
        try:
            gtype = payload.get("type")
            if gtype == "FeatureCollection":
                bounds_list = [
                    _shapely_shape(feat["geometry"]).bounds
                    for feat in payload.get("features", [])
                    if isinstance(feat, dict) and feat.get("geometry")
                ]
                if bounds_list:
                    w = min(b[0] for b in bounds_list)
                    s = min(b[1] for b in bounds_list)
                    e = max(b[2] for b in bounds_list)
                    n = max(b[3] for b in bounds_list)
                    return (round(float(w), 4), round(float(s), 4), round(float(e), 4), round(float(n), 4))
            elif gtype == "Feature" and payload.get("geometry"):
                b = _shapely_shape(payload["geometry"]).bounds
                return (round(float(b[0]), 4), round(float(b[1]), 4), round(float(b[2]), 4), round(float(b[3]), 4))
            elif "coordinates" in payload:
                b = _shapely_shape(payload).bounds
                return (round(float(b[0]), 4), round(float(b[1]), 4), round(float(b[2]), 4), round(float(b[3]), 4))
        except Exception:
            pass

    lons: list[float] = []
    lats: list[float] = []

    def _walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[0], (int, float)) and isinstance(node[1], (int, float)):
                lons.append(float(node[0]))
                lats.append(float(node[1]))
            else:
                for item in node:
                    _walk(item)
        elif isinstance(node, dict):
            for val in node.values():
                _walk(val)

    _walk(payload)
    if not lons or not lats:
        return None
    return (round(min(lons), 4), round(min(lats), 4), round(max(lons), 4), round(max(lats), 4))


def _apply_bbox_to_state(bbox: tuple[float, float, float, float]) -> None:
    west, south, east, north = (round(float(v), 4) for v in bbox)
    st.session_state["bbox_west"] = west
    st.session_state["bbox_south"] = south
    st.session_state["bbox_east"] = east
    st.session_state["bbox_north"] = north
    st.session_state["applied_bbox"] = (west, south, east, north)


def _init_session_state() -> None:
    default_region = "cordoba_argentina"
    if "selected_region" not in st.session_state:
        st.session_state["selected_region"] = default_region
    if "last_preset" not in st.session_state:
        st.session_state["last_preset"] = default_region
    if "bbox_west" not in st.session_state:
        _apply_bbox_to_state(REGION_PRESETS[default_region]["bbox"])
    if "applied_bbox" not in st.session_state:
        default_bbox = REGION_PRESETS[default_region]["bbox"]
        st.session_state["applied_bbox"] = tuple(round(float(v), 4) for v in default_bbox)
    if "pending_bbox" in st.session_state and st.session_state["pending_bbox"] is not None:
        _apply_bbox_to_state(st.session_state.pop("pending_bbox"))
    if "api_modis_df" not in st.session_state:
        st.session_state["api_modis_df"] = pd.DataFrame()
    if "api_viirs_df" not in st.session_state:
        st.session_state["api_viirs_df"] = pd.DataFrame()
    if "api_availability_df" not in st.session_state:
        st.session_state["api_availability_df"] = pd.DataFrame()
    if "api_availability_attempted" not in st.session_state:
        st.session_state["api_availability_attempted"] = False
    if "api_fetch_meta" not in st.session_state:
        st.session_state["api_fetch_meta"] = None
    if "quick_location" not in st.session_state:
        st.session_state["quick_location"] = "Custom Coordinates (Manual Input)"
    if "last_quick_location" not in st.session_state:
        st.session_state["last_quick_location"] = "Custom Coordinates (Manual Input)"
    if "api_failed_windows" not in st.session_state:
        st.session_state["api_failed_windows"] = []
    if "last_geojson_sig" not in st.session_state:
        st.session_state["last_geojson_sig"] = ""


def _estimate_zoom_for_bbox(bbox: tuple[float, float, float, float]) -> int:
    west, south, east, north = bbox
    span = max(abs(east - west), abs(north - south), 0.05)
    if span <= 0.6:
        return 9
    if span <= 1.5:
        return 8
    if span <= 3.5:
        return 7
    if span <= 7.5:
        return 6
    if span <= 15.0:
        return 5
    return 4


def _compact_status(status: str) -> str:
    """Shorten climatology status labels so 4-column KPI cards do not clip."""
    mapping = {
        "Critical Anomaly (Robust Z > +1.5)": "Critical (Robust Z>+1.5)",
        "Critical Anomaly (> +1.5σ)": "Critical (> +1.5σ)",
        "Extreme Anomaly (> +1.5σ)": "Critical (> +1.5σ)",
        "Above Normal (> P75)": "Above Normal (>P75)",
        "Below Normal (< P25)": "Below Normal (<P25)",
        "Normal (P25–P75)": "Normal (P25–P75)",
        "Insufficient Data (<8 yrs)": "Low n (<8 yrs)",
        "No Data": "No Data",
    }
    return mapping.get(status, status)


# ---------------------------------------------------------------------------
# Plotly Chart Builders
# ---------------------------------------------------------------------------
def _make_calendar_heatmap(
    calendar_df: pd.DataFrame,
    colorscale: str,
    metric_label: str,
    color_scale_mode: str = "Robust (P95 Clamped)",
) -> go.Figure:
    years = [str(y) for y in calendar_df.index]
    raw_z = calendar_df.to_numpy(dtype=float)
    show_text = len(years) <= 30
    cb_unit = "MW" if ("frp" in metric_label.lower() or "mw" in metric_label.lower()) else "Cells"
    clean_ints = np.nan_to_num(np.round(raw_z, 0), nan=0.0).astype(int).astype(str)
    text_matrix = np.where(np.isnan(raw_z), "", clean_ints) if show_text else None

    pos_vals = raw_z[np.isfinite(raw_z) & (raw_z > 0)]
    if color_scale_mode == "Logarithmic (log1p)":
        z_plot = np.where(np.isfinite(raw_z), np.log1p(np.maximum(raw_z, 0.0)), np.nan)
        zmin: Optional[float] = 0.0
        zmax: Optional[float] = None
        cb_title = f"log1p({cb_unit})"
    elif color_scale_mode == "Linear (Min–Max)":
        z_plot = raw_z
        zmin = None
        zmax = None
        cb_title = cb_unit
    else:  # "Robust (P95 Clamped)"
        z_plot = raw_z
        zmin = 0.0
        if pos_vals.size > 0:
            p95 = float(np.nanpercentile(pos_vals, 95))
            zmax = p95 if p95 > 0 else float(np.nanmax(pos_vals))
        else:
            zmax = None
        cb_title = f"{cb_unit} (P95)"

    heatmap = go.Heatmap(
        z=z_plot,
        zmin=zmin,
        zmax=zmax,
        x=list(calendar_df.columns),
        y=years,
        colorscale=colorscale,
        colorbar=dict(title=dict(text=cb_title, side="right"), thickness=14, len=0.9),
        customdata=raw_z,
        hovertemplate="Year: %{y}<br>Month: %{x}<br>" + metric_label + ": %{customdata:,.1f}<extra></extra>",
        text=text_matrix,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(size=10),
    )
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=dict(
            text=f"Burning Activity Calendar Heatmap (Year × Month) — {metric_label} (Gray = No Data)",
            font=dict(size=14),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        plot_bgcolor="#1e293b",
        dragmode=False,
        xaxis=dict(title="Calendar Month", fixedrange=True),
        yaxis=dict(title="Year", autorange="reversed", fixedrange=True),
        height=max(420, min(720, 26 * max(len(years), 6) + 120)),
        margin=dict(l=55, r=30, t=65, b=45),
    )
    return fig


def _make_seasonal_profile_chart(
    climatology_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    target_year: int,
    metric_label: str,
    accent_color: str,
) -> go.Figure:
    year_df = (
        anomalies_df[anomalies_df["year"] == target_year]
        .sort_values("month")
        .reset_index(drop=True)
    )
    clim_band = climatology_df.dropna(subset=["p25", "p75", "median"])
    months_band = clim_band["month_name"].tolist()
    months = climatology_df["month_name"].tolist()

    fig = go.Figure()
    # P25-P75 interquartile band
    if months_band:
        fig.add_trace(
            go.Scatter(
                x=months_band + months_band[::-1],
                y=clim_band["p75"].tolist() + clim_band["p25"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(100, 149, 237, 0.22)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Historical IQR (P25–P75)",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=climatology_df["median"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2.5, dash="dash"),
            name="Historical Median",
        )
    )
    if not year_df.empty:
        fig.add_trace(
            go.Scatter(
                x=year_df["month_name"],
                y=year_df["value"],
                mode="lines+markers",
                line=dict(color=accent_color, width=3.5),
                marker=dict(size=9),
                name=f"Selected Year ({target_year})",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Seasonal Profile ({target_year} vs. Baseline)",
            font=dict(size=13),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        dragmode=False,
        xaxis=dict(title="Month", fixedrange=True),
        yaxis=dict(title=metric_label, fixedrange=True),
        hovermode="x unified",
        height=430,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=68, b=90),
    )
    return fig


def _make_anomaly_bar_chart(
    anomalies_df: pd.DataFrame,
    target_year: int,
    mode: str,
) -> go.Figure:
    year_df = (
        anomalies_df[anomalies_df["year"] == target_year]
        .sort_values("month")
        .reset_index(drop=True)
    )
    if mode == "% Deviation from Median":
        col = "anomaly_pct"
        y_title = "Deviation from Median (%)"
    elif mode == "Z-score (σ)":
        col = "z_score"
        y_title = "Classical Z-score (σ)"
    else:
        col = "robust_z_score" if "robust_z_score" in year_df.columns else "z_score"
        y_title = "Robust Z-score (0.7413·IQR)"

    colors = []
    text_vals = []
    for _, row in year_df.iterrows():
        status_str = str(row.get("anomaly_status", ""))
        v = row.get(col)
        if status_str == "No Data" or pd.isna(row.get("value")) or pd.isna(v):
            colors.append("#475569")
            text_vals.append("No Data")
            continue
        rz = float(row.get("robust_z_score", row.get("z_score", 0.0)) or 0.0)
        val = float(row["value"])
        p75 = float(row.get("p75", np.nan))
        p25 = float(row.get("p25", np.nan))
        if status_str.startswith("Critical") or (rz > 1.5 and pd.notna(p75) and val > p75):
            colors.append("#d62728")
        elif pd.notna(p75) and val > p75:
            colors.append("#ff7f0e")
        elif pd.notna(p25) and val < p25:
            colors.append("#6c757d")
        else:
            colors.append("#2ca02c")
        text_vals.append(f"{float(v):+.1f}")

    y_vals = pd.to_numeric(year_df[col], errors="coerce")
    fig = go.Figure(
        data=[
            go.Bar(
                x=year_df["month_name"],
                y=y_vals,
                marker_color=colors,
                text=text_vals,
                textposition="outside",
                customdata=np.stack(
                    [year_df["value"], year_df["median"], year_df["anomaly_status"]],
                    axis=-1,
                ),
                hovertemplate=(
                    "Month: %{x}<br>"
                    + y_title
                    + ": %{y:+.2f}<br>"
                    + "Actual: %{customdata[0]:,.1f} | LOO Median: %{customdata[1]:,.1f}<br>"
                    + "Status: %{customdata[2]}<extra></extra>"
                ),
            )
        ]
    )
    fig.add_hline(y=0, line_width=1.2, line_color="#64748b")
    if mode != "% Deviation from Median":
        fig.add_hline(
            y=1.5,
            line_dash="dot",
            line_color="#d62728",
            annotation_text="Critical Threshold (+1.5)",
            annotation_font_color="#f8fafc",
        )
    fig.update_layout(
        title=dict(
            text=f"Monthly Fire Anomalies ({target_year}, LOO Baseline): {mode}",
            font=dict(size=13),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        dragmode=False,
        xaxis=dict(title="Month", fixedrange=True),
        yaxis=dict(title=y_title, fixedrange=True),
        height=430,
        margin=dict(l=50, r=20, t=68, b=90),
    )
    return fig


def _make_raw_vs_harmonized_chart(
    monthly_df: pd.DataFrame,
    granularity: str,
    use_frp: bool,
) -> go.Figure:
    if granularity == "Annual Total":
        obs_df = (
            monthly_df[monthly_df["data_status"] != "no_data"]
            if "data_status" in monthly_df.columns
            else monthly_df
        )
        obs_df = obs_df[obs_df["harmonized_hotspots"].notna()] if "harmonized_hotspots" in obs_df.columns else obs_df
        current_yr = date.today().year
        month_counts = obs_df.groupby("year")["month"].nunique()
        full_years = [int(y) for y, cnt in month_counts.items() if int(y) < current_yr or int(cnt) == 12]
        df_src = obs_df[obs_df["year"].isin(full_years)] if full_years else obs_df
        df = df_src.groupby("year", as_index=False).sum(numeric_only=True, min_count=1)
        x_col = "year"
        transition_x: object = 2012
    else:
        df = monthly_df.sort_values(["year", "month"]).copy()
        x_col = "date_label"
        transition_x = "2012-03"

    raw_col = "raw_total_frp" if use_frp else "raw_total_points"
    modis_col = "modis_frp_sum" if use_frp else "modis_active_cells"
    viirs_col = "viirs_frp_sum" if use_frp else "viirs_active_cells"
    harm_col = "harmonized_frp" if use_frp else "harmonized_hotspots"
    y_title = "Fire Radiative Power (FRP, MW)" if use_frp else "Hot Spots / Detections Count"

    fig = go.Figure()
    if not use_frp:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[raw_col],
                mode="lines+markers",
                name="BEFORE: Raw Point Sum (MODIS + VIIRS)",
                line=dict(color="#d62728", width=2.2, dash="dash"),
                marker=dict(size=5),
            )
        )
        post_mask = df["year"] >= 2012
        fig.add_trace(
            go.Scatter(
                x=df.loc[post_mask, x_col],
                y=df.loc[post_mask, viirs_col],
                mode="lines",
                name="VIIRS 0.01° Grid (Uncalibrated, without α)",
                line=dict(color="#ff7f0e", width=2.0, dash="dot"),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[raw_col],
                mode="lines",
                name="BEFORE: Raw FRP Sum (MODIS + VIIRS)",
                line=dict(color="#d62728", width=2.2, dash="dash"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[modis_col],
            mode="lines",
            name="MODIS 1 km Reference Series",
            line=dict(color="#17becf", width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[harm_col],
            mode="lines+markers",
            name="AFTER: Harmonized Series (MODIS + α·VIIRS)",
            line=dict(color="#2ca02c", width=3.5),
            marker=dict(size=6),
        )
    )

    if str(transition_x) in set(df[x_col].astype(str)):
        fig.add_vline(
            x=transition_x,
            line_width=2,
            line_dash="dashdot",
            line_color="#cbd5e1",
        )
        fig.add_annotation(
            x=transition_x,
            y=1.02,
            yref="paper",
            text="VIIRS Launch (2012-03)",
            showarrow=False,
            font=dict(size=12, color="#f8fafc"),
            bgcolor="rgba(15, 23, 42, 0.82)",
            bordercolor="#94a3b8",
            borderpad=3,
        )

    fig.update_layout(
        title=dict(
            text="Before vs. After Harmonization: Eliminating the 2012 Sensor Step Jump",
            font=dict(size=14),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        dragmode=False,
        xaxis=dict(title="Period", fixedrange=True),
        yaxis=dict(title=y_title, fixedrange=True),
        hovermode="x unified",
        height=470,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=55, r=25, t=65, b=95),
    )
    return fig


def _make_live_daily_chart(
    gridded_df: pd.DataFrame,
    use_frp: bool,
    harmonized_cells_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Build a daily breakdown chart for short Live API / <2-year operational windows."""
    fig = go.Figure()
    if gridded_df.empty:
        fig.update_layout(
            title="Operational Window: No Dated Detections",
            dragmode=False,
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
        )
        return fig

    daily = build_daily_sensor_breakdown(
        gridded_df=gridded_df,
        harmonized_cells_df=harmonized_cells_df,
    )

    if not use_frp:
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["raw_points"],
                name="BEFORE: Raw Detections",
                marker_color="#d62728",
            )
        )
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["modis_cells"],
                name="MODIS 0.01° Cells",
                marker_color="#17becf",
            )
        )
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["viirs_cells"],
                name="VIIRS 0.01° Grid (without α)",
                marker_color="#ff7f0e",
            )
        )
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["harmonized_cells"],
                name="AFTER: Harmonized (α·VIIRS)",
                marker_color="#2ca02c",
            )
        )
        y_title = "Daily Hot Spots / Detections"
    else:
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["modis_frp"],
                name="MODIS FRP (MW)",
                marker_color="#17becf",
            )
        )
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["viirs_frp"],
                name="VIIRS FRP (Native MW)",
                marker_color="#ff7f0e",
            )
        )
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["harmonized_frp"],
                name="AFTER: Harmonized FRP (MW)",
                marker_color="#2ca02c",
            )
        )
        y_title = "Fire Radiative Power (MW)"

    fig.update_layout(
        title=dict(
            text="Short-Window Mode (< 2 years): Daily Sensor & Harmonization Breakdown",
            font=dict(size=14),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        dragmode=False,
        xaxis=dict(title="Acquisition Date (UTC)", fixedrange=True),
        yaxis=dict(title=y_title, fixedrange=True),
        barmode="group",
        height=390,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=55, r=25, t=65, b=85),
    )
    return fig


def _render_daily_breakdown_table(daily_df: pd.DataFrame) -> None:
    """Display per-day, per-sensor counts and separate MODIS/VIIRS FRP table."""
    if daily_df is None or daily_df.empty:
        st.info("No daily observations available in this window.")
        return
    view_df = daily_df[
        [
            "acq_date",
            "modis_raw_points",
            "modis_cells",
            "modis_frp",
            "viirs_raw_points",
            "viirs_cells",
            "viirs_frp",
            "harmonized_cells",
            "harmonized_frp",
        ]
    ].copy()
    view_df["modis_frp"] = view_df["modis_frp"].round(1)
    view_df["viirs_frp"] = view_df["viirs_frp"].round(1)
    view_df["harmonized_cells"] = view_df["harmonized_cells"].round(2)
    view_df["harmonized_frp"] = view_df["harmonized_frp"].round(1)
    view_df = view_df.rename(
        columns={
            "acq_date": "Date (UTC)",
            "modis_raw_points": "MODIS Raw Pts",
            "modis_cells": "MODIS 0.01° Cells",
            "modis_frp": "MODIS FRP (MW)",
            "viirs_raw_points": "VIIRS Raw Pts",
            "viirs_cells": "VIIRS 0.01° Cells",
            "viirs_frp": "VIIRS FRP (MW)",
            "harmonized_cells": "Harmonized 0.01° Cells",
            "harmonized_frp": "Harmonized FRP (MW)",
        }
    )
    st.dataframe(view_df, hide_index=True, width="stretch")


def _make_overlap_scatter(
    overlap_df: pd.DataFrame,
    tm: dict[str, float],
) -> go.Figure:
    valid = overlap_df[
        (overlap_df["modis_active_cells"] > 0) & (overlap_df["viirs_active_cells"] > 0)
    ].copy()
    fig = go.Figure()
    if valid.empty:
        fig.update_layout(
            title="Insufficient Paired Observations (2012–2025) for Regression",
            dragmode=False,
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=valid["viirs_active_cells"],
            y=valid["modis_active_cells"],
            mode="markers",
            name="Overlap Months (2012–2025)",
            marker=dict(
                size=8,
                color=valid["year"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Year", thickness=12, len=0.85),
                opacity=0.82,
            ),
            text=[f"{int(y)}-{int(m):02d}" for y, m in zip(valid["year"], valid["month"])],
            hovertemplate="Period: %{text}<br>VIIRS Cells: %{x:,.0f}<br>MODIS Cells: %{y:,.0f}<extra></extra>",
        )
    )

    x_line = np.linspace(valid["viirs_active_cells"].min(), valid["viirs_active_cells"].max(), 50)
    slope = tm.get("ols_slope", 1.0)
    intercept = tm.get("ols_intercept", 0.0)
    y_line = slope * x_line + intercept
    intercept_sign = "+" if intercept >= 0 else "-"
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"OLS: y = {slope:.3f}x {intercept_sign} {abs(intercept):.1f}",
            line=dict(color="#d62728", width=2.5),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"MODIS–VIIRS Overlap (2012–2025): "
                f"R²={tm.get('overlap_r2', 0.0):.4f}, r={tm.get('overlap_pearson_r', 0.0):.4f}"
            ),
            font=dict(size=13),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        dragmode=False,
        xaxis=dict(title="VIIRS Active 0.01° Cells (count/month)", fixedrange=True),
        yaxis=dict(title="MODIS Active 0.01° Cells (count/month)", fixedrange=True),
        height=430,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=68, b=90),
    )
    return fig


def _make_alpha_bar_chart(calibration_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for dn, label, color in [
        ("D", "Daytime (D)", "#ff7f0e"),
        ("N", "Nighttime (N)", "#1f77b4"),
    ]:
        sub = calibration_df[calibration_df["daynight"] == dn].sort_values("month")
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[MONTH_LABELS[m - 1] for m in sub["month"]],
                y=sub["alpha_cells"],
                name=label,
                marker_color=color,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(sub["alpha_ci_high"] - sub["alpha_cells"]).clip(lower=0.0).fillna(0.0),
                    arrayminus=(sub["alpha_cells"] - sub["alpha_ci_low"]).clip(lower=0.0).fillna(0.0),
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Seasonal Calibration Factors α(m, D/N) with 95% Bootstrap CI",
            font=dict(size=13),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        dragmode=False,
        xaxis=dict(title="Month", fixedrange=True),
        yaxis=dict(title="Scaling Factor α (VIIRS → MODIS)", fixedrange=True),
        barmode="group",
        height=430,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=68, b=90),
    )
    return fig


def _make_critical_windows_chart(crit_df: pd.DataFrame) -> go.Figure:
    """Plot 52-week empirical exceedance probability P(Weekly Activity > P75)."""
    fig = go.Figure()
    if crit_df.empty:
        return fig

    pct_exc = crit_df["p_exceedance"].to_numpy(dtype=float) * 100.0
    colors = ["#ef4444" if bool(c) else "#38bdf8" for c in crit_df["is_critical"]]
    threshold_p75 = float(crit_df["threshold_p75"].iloc[0])

    fig.add_trace(
        go.Bar(
            x=crit_df["week_label"],
            y=pct_exc,
            marker_color=colors,
            name="P(Weekly > P75)",
            customdata=np.stack(
                [crit_df["mean_weekly"], crit_df["max_weekly"], crit_df["n_years"]],
                axis=-1,
            ),
            hovertemplate=(
                "%{x}<br>Exceedance Prob (> P75): %{y:.1f}%<br>"
                + f"Region P75 Threshold: {threshold_p75:,.1f} cells/wk<br>"
                + "Weekly Mean: %{customdata[0]:,.1f} | Historical Max: %{customdata[1]:,.1f} "
                + "(n=%{customdata[2]:.0f} yrs)<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=50.0,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="Critical Fire Window (>= 50% exceedance)",
        annotation_font_color="#f8fafc",
    )
    fig.update_layout(
        title=dict(
            text=f"52-Week Critical Fire Risk Windows — P(Weekly Harmonized Cells > P75 = {threshold_p75:,.1f})",
            font=dict(size=13),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        dragmode=False,
        xaxis=dict(title="Climatological Week (7-day window)", tickangle=-45, nticks=18, fixedrange=True),
        yaxis=dict(title="Exceedance Probability (%)", range=[0, 108], fixedrange=True),
        height=340,
        margin=dict(l=50, r=20, t=58, b=80),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Folium Geospatial Map Builder
# ---------------------------------------------------------------------------
class _SafeLayerControl(folium.LayerControl):
    """LayerControl using `var` and viewport-aware Leaflet resize hooks for tabs."""

    def __init__(
        self,
        bbox_bounds: Optional[list[list[float]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bbox_bounds = bbox_bounds or [[-85.0, -180.0], [85.0, 180.0]]

    _template = Template(
        """
        {% macro script(this,kwargs) %}
            var {{ this.get_name() }}_layers = {
                base_layers : {
                    {%- for key, val in this.base_layers.items() %}
                    {{ key|tojson }} : {{val}},
                    {%- endfor %}
                },
                overlays :  {
                    {%- for key, val in this.overlays.items() %}
                    {{ key|tojson }} : {{val}},
                    {%- endfor %}
                },
            };
            var {{ this.get_name() }} = L.control.layers(
                {{ this.get_name() }}_layers.base_layers,
                {{ this.get_name() }}_layers.overlays,
                {{ this.options|tojavascript }}
            ).addTo({{this._parent.get_name()}});

            {%- if this.draggable %}
            new L.Draggable({{ this.get_name() }}.getContainer()).enable();
            {%- endif %}

            (function() {
                var _map = {{ this._parent.get_name() }};
                var _bboxBounds = L.latLngBounds({{ this.bbox_bounds|tojson }});
                var _lastWidth = 0;
                var _lastHeight = 0;
                var _refreshMapSize = function() {
                    var el = _map.getContainer();
                    if (!el) return;
                    var w = el.offsetWidth;
                    var h = el.offsetHeight;
                    if (w > 0 && h > 0 && (w !== _lastWidth || h !== _lastHeight)) {
                        var firstShow = (_lastWidth === 0);
                        _lastWidth = w;
                        _lastHeight = h;
                        _map.invalidateSize(false);
                        _map.options.zoomSnap = 0.05;
                        _map.options.zoomDelta = 0.5;
                        _map.options.bounceAtZoomLimits = false;

                        var minZoomForHeight = Math.ceil(Math.log2(h / 256) * 20) / 20;
                        minZoomForHeight = Math.max(1.5, minZoomForHeight);
                        _map.setMinZoom(minZoomForHeight);
                        _map.setMaxBounds([[-85, -1000000], [85, 1000000]]);

                        if (firstShow) {
                            var fitZ = _map.getBoundsZoom(_bboxBounds, false, L.point(16, 16));
                            var targetZoom = Math.max(fitZ, minZoomForHeight);
                            _map.setView(_bboxBounds.getCenter(), targetZoom, {animate: false});
                        } else if (_map.getZoom() < minZoomForHeight) {
                            _map.setZoom(minZoomForHeight, {animate: false});
                        }
                    }
                };
                if (typeof ResizeObserver !== "undefined") {
                    new ResizeObserver(_refreshMapSize).observe(_map.getContainer());
                }
                window.addEventListener("resize", _refreshMapSize);
                setInterval(_refreshMapSize, 250);
            })();
        {% endmacro %}
        """
    )
def _build_folium_map(
    gridded_df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    max_points: int = 300,
) -> folium.Map:
    west, south, east, north = bbox
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=_estimate_zoom_for_bbox(bbox),
        tiles=None,
        zoom_control=False,
        world_copy_jump=True,
        no_wrap=False,
        min_zoom=2,
        max_bounds=True,
        max_bounds_viscosity=1.0,
        min_lat=-85.0,
        max_lat=85.0,
        min_lon=-1000000.0,
        max_lon=1000000.0,
    )
    fmap.get_root().header.add_child(
        folium.Element(
            "<style>"
            "html, body { margin: 0 !important; padding: 0 !important; "
            "width: 100% !important; height: 100% !important; overflow: hidden !important; } "
            ".leaflet-container { width: 100% !important; height: 100% !important; background: #aad3df !important; }"
            "</style>"
        )
    )
    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        no_wrap=False,
    ).add_to(fmap)

    # Active Bounding Box polygon
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#21918c",
        weight=2.5,
        fill=True,
        fill_opacity=0.05,
        tooltip=f"Active BBox: [{west:.2f}, {south:.2f}, {east:.2f}, {north:.2f}]",
    ).add_to(fmap)

    # External OGC WMS Layers (NASA GIBS FIRMS + CONAE Argentina)
    folium.WmsTileLayer(
        url="https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi",
        layers="MODIS_Terra_Thermal_Anomalies_All",
        fmt="image/png",
        transparent=True,
        name="NASA GIBS WMS — MODIS Thermal Anomalies",
        attr="NASA EOSDIS GIBS / FIRMS",
        overlay=True,
        control=True,
        show=False,
    ).add_to(fmap)

    folium.WmsTileLayer(
        url=CONAE_WMS_BASE_URL,
        layers="FocosDeCalor:focos_viirs",
        fmt="image/png",
        transparent=True,
        name="CONAE OGC WMS — Focos de Calor",
        attr="CONAE Argentina",
        overlay=True,
        control=True,
        show=False,
    ).add_to(fmap)

    if not gridded_df.empty:
        sample_df = (
            gridded_df.nlargest(max_points, "frp_sum")
            if len(gridded_df) > max_points
            else gridded_df
        )

        modis_fg = folium.FeatureGroup(name="MODIS 1 km Cells", show=True)
        viirs_fg = folium.FeatureGroup(name="VIIRS 375 m → 1 km Cells", show=True)

        for row in sample_df.itertuples(index=False):
            is_modis = row.sensor_group == "MODIS"
            target_group = modis_fg if is_modis else viirs_fg
            color = "#0077b6" if is_modis else "#e85d04"
            radius = float(np.clip(2.5 + np.log1p(row.frp_sum) * 0.9, 3.0, 9.0))
            popup_html = (
                f"<b>Sensor:</b> {row.sensor_group} ({row.daynight})<br>"
                f"<b>Date:</b> {row.acq_date}<br>"
                f"<b>Cell Center:</b> {row.cell_center_lat:.3f}, {row.cell_center_lon:.3f}<br>"
                f"<b>Raw Detections in Cell:</b> {int(row.raw_point_count)}<br>"
                f"<b>FRP (Sum):</b> {row.frp_sum:.1f} MW"
            )
            folium.CircleMarker(
                location=[row.cell_center_lat, row.cell_center_lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
                weight=1,
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(target_group)

        modis_fg.add_to(fmap)
        viirs_fg.add_to(fmap)

        heat_points = [
            [float(r.cell_center_lat), float(r.cell_center_lon), float(np.log1p(r.frp_sum))]
            for r in gridded_df.itertuples(index=False)
        ]
        if heat_points:
            HeatMap(
                heat_points,
                name="FRP Thermal Density",
                radius=14,
                blur=16,
                min_opacity=0.35,
                show=True,
            ).add_to(fmap)

    _SafeLayerControl(bbox_bounds=[[south, west], [north, east]], collapsed=False).add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session_state()
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        [data-testid="stToolbarActions"],
        [data-testid="stAppDeployButton"],
        .stAppDeployButton,
        [data-testid="stMainMenu"],
        #MainMenu,
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        footer {
            display: none !important;
        }
        [data-testid="stExpandSidebarButton"],
        [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"],
        section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child,
        section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarContent"] {
            min-width: 430px !important;
            width: 430px !important;
            max-width: 430px !important;
        }
        @media (max-width: 640px) {
            section[data-testid="stSidebar"][aria-expanded="true"],
            section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child,
            section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarContent"] {
                min-width: 100% !important;
                width: 100% !important;
                max-width: 100% !important;
            }
        }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-baseweb="select"] span,
        section[data-testid="stSidebar"] [data-baseweb="select"] div {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            height: auto !important;
            min-height: 2.2rem !important;
        }
        [data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 14px 16px;
            min-width: 200px;
        }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }
        [data-testid="stColumn"],
        [data-testid="column"] {
            min-width: 200px !important;
        }
        [data-testid="stSidebar"] [data-testid="stColumn"],
        [data-testid="stSidebar"] [data-testid="column"] {
            min-width: 0 !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #cbd5e1;
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        [data-testid="stMetricDelta"] > div,
        [data-testid="stMetricDelta"] div,
        [data-testid="stMetricLabel"] p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔥 MODIS & VIIRS Hot Spots Harmonizer — Burning Activity Calendar")
    st.caption(
        "NASA Space Apps Challenge 2026 (NASA & CONAE) · "
        "Removing the 2012 MODIS (1 km) → VIIRS (375 m) sensor discontinuity to produce "
        "a seamless, climatology-ready 2001–2026 fire record."
    )

    with st.expander(
        "📖 What do these metrics mean? (Quick Guide to FRP, α, Z-score & 0.01° Grid)",
        expanded=False,
    ):
        st.markdown(
            """
- **Active Fire Cell (`0.01° × 0.01° ≈ 1 km²`)**: VIIRS (`375 m`) resolves sub-pixel fire clusters that MODIS (`1 km`) reports as a single hotspot, inflating raw detection counts after March 2012. Gridding merges co-located detections within each `0.01°` grid cell per sensor and overpass (`Day`/`Night`) onto a standardized 1 km scale.
- **FRP (Fire Radiative Power, MW)**: Instantaneous radiative heat release measured in megawatts (`MW`). Total FRP reflects fire intensity and fuel consumption rate rather than raw pixel counts.
- **Calibration Factor `α` (VIIRS → MODIS)**: Monthly day/night ratio derived from the `2012–2025` MODIS–VIIRS overlap period (e.g. Córdoba real archive: `Day ~0.64`, `Night ~0.17`). Nighttime `α` is lower because VIIRS detects cooler, smaller nighttime smoldering fires invisible to MODIS at `1 km`.
- **Z-score (`σ`) & IQR (`P25–P75`)**: Historical climatology (`2001–2025`) defines normal seasonal activity (`P25–P75` interquartile band). Values above `P75` are **Above Normal**, values below `P25` are **Below Normal**, and `Z > +1.5σ` flags a **Critical Anomaly**.
            """
        )

    # -----------------------------------------------------------------------
    # Sidebar Controls
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Control Panel")

        ref_mode_label = "📌 Reference Case — Córdoba, Argentina (2001–2025)"
        explore_mode_label = "🌐 Explore Any Region (Live API / Custom CSV)"
        real_source_label = "Real NASA FIRMS Archive — Córdoba (2001–2025)"
        synth_source_label = "Synthetic Validation Benchmark (2001–2026)"
        api_source_label = "Live NASA FIRMS Area API"
        upload_source_label = "Upload CSV (FIRMS Archive)"

        operating_mode = st.radio(
            "Operating Mode",
            options=[ref_mode_label, explore_mode_label],
            index=0,
            help=(
                "Use Reference Case to explore the validated 2001–2025 Córdoba archive or synthetic benchmarks, "
                "or switch to Explore Any Region for Live NASA FIRMS API / custom CSVs anywhere on Earth."
            ),
        )
        is_explore_mode = operating_mode == explore_mode_label

        if not is_explore_mode:
            source_mode = st.radio(
                "Data Source",
                options=[real_source_label, synth_source_label],
                index=0,
                help=(
                    "Inspect the official 2001–2025 NASA FIRMS satellite archive for Córdoba (MODIS C6.1 + VIIRS C2) "
                    "or the 3-biome synthetic ground-truth recovery benchmark."
                ),
            )
        else:
            source_mode = st.radio(
                "Data Source",
                options=[api_source_label, upload_source_label],
                index=0,
                help=(
                    "Query live near real-time detections via the NASA LANCE FIRMS Area API "
                    "or upload custom MODIS/VIIRS FIRMS CSV archives."
                ),
            )

        is_real_cordoba_mode = (not is_explore_mode) and (source_mode == real_source_label)
        if is_real_cordoba_mode and st.session_state.get("last_source_mode") != source_mode:
            st.session_state["selected_region"] = "cordoba_argentina"
            st.session_state["last_preset"] = "cordoba_argentina"
            _apply_bbox_to_state(REGION_PRESETS["cordoba_argentina"]["bbox"])
        st.session_state["last_source_mode"] = source_mode

        st.subheader("🌍 Region & Bounding Box")
        quick_loc = "Custom Coordinates (Manual Input)"
        region_key = st.session_state.get("selected_region", "cordoba_argentina")

        if not is_explore_mode:
            preset_keys = list(REGION_PRESETS.keys())
            region_key = st.selectbox(
                "Region Preset",
                options=preset_keys,
                format_func=lambda k: REGION_PRESETS[k]["name"],
                key="selected_region",
                disabled=is_real_cordoba_mode,
                help=(
                    "Locked to Córdoba Province when viewing the Real NASA FIRMS Córdoba archive; "
                    "all 3 regional biomes are unlocked in Synthetic Validation Benchmark mode."
                ),
            )
            if region_key != st.session_state["last_preset"]:
                st.session_state["last_preset"] = region_key
                _apply_bbox_to_state(REGION_PRESETS[region_key]["bbox"])
        else:
            quick_loc = st.selectbox(
                "Quick Locations (Helper BBox)",
                options=list(QUICK_LOCATIONS.keys()),
                key="quick_location",
                help=(
                    "Populate the Bounding Box with coordinates for curated global regions, "
                    "or select Custom Coordinates (Manual Input) to enter any bounds."
                ),
            )
            if quick_loc != st.session_state["last_quick_location"]:
                st.session_state["last_quick_location"] = quick_loc
                helper_bbox = QUICK_LOCATIONS[quick_loc]
                if helper_bbox is not None:
                    _apply_bbox_to_state(helper_bbox)

            geojson_file = st.file_uploader(
                "Upload AOI Polygon/BBox (.geojson / .json)",
                type=["geojson", "json"],
                help="Optional: upload a GeoJSON Polygon, MultiPolygon, Feature, or FeatureCollection to set the Bounding Box automatically.",
            )
            if geojson_file is not None:
                gj_sig = f"{geojson_file.name}:{geojson_file.size}"
                if gj_sig != st.session_state.get("last_geojson_sig", ""):
                    try:
                        payload = json.loads(geojson_file.getvalue().decode("utf-8"))
                        gj_bbox = _extract_bbox_from_geojson(payload)
                        if gj_bbox is not None:
                            st.session_state["last_geojson_sig"] = gj_sig
                            _apply_bbox_to_state(gj_bbox)
                        else:
                            st.warning("Uploaded GeoJSON contained no valid coordinates.")
                    except Exception as exc:
                        st.error(f"Could not parse GeoJSON AOI: {exc}")

        with st.form("bbox_coords_form", border=False):
            col_w, col_s = st.columns(2)
            west = col_w.number_input(
                "Min Lon (West)",
                step=0.1,
                format="%.3f",
                key="bbox_west",
                help="Western longitude bound in decimal degrees (-180 to 180).",
            )
            south = col_s.number_input(
                "Min Lat (South)",
                step=0.1,
                format="%.3f",
                key="bbox_south",
                help="Southern latitude bound in decimal degrees (-90 to 90).",
            )
            col_e, col_n = st.columns(2)
            east = col_e.number_input(
                "Max Lon (East)",
                step=0.1,
                format="%.3f",
                key="bbox_east",
                help="Eastern longitude bound in decimal degrees (-180 to 180).",
            )
            north = col_n.number_input(
                "Max Lat (North)",
                step=0.1,
                format="%.3f",
                key="bbox_north",
                help="Northern latitude bound in decimal degrees (-90 to 90).",
            )
            apply_bbox_clicked = st.form_submit_button(
                "🎯 Apply Coordinates & Update Map",
                type="primary",
                width="stretch",
            )

        if apply_bbox_clicked:
            st.session_state["applied_bbox"] = (
                round(float(west), 4),
                round(float(south), 4),
                round(float(east), 4),
                round(float(north), 4),
            )

        if not is_explore_mode:
            if st.button("Reset BBox to Preset", width="stretch"):
                st.session_state["pending_bbox"] = REGION_PRESETS[region_key]["bbox"]
                st.rerun()

        active_bbox = st.session_state["applied_bbox"]
        firms_compare_url = build_firms_map_url(active_bbox, zoom=_estimate_zoom_for_bbox(active_bbox))

        st.link_button(
            "🌐 Compare in Official NASA FIRMS Fire Map",
            firms_compare_url,
            width="stretch",
            help="Open the current Bounding Box center on NASA's official FIRMS Global Fire Map.",
        )
        if not is_explore_mode:
            st.link_button(
                "🛰️ Open Official CONAE Geoportal",
                CONAE_FOCOS_URL,
                width="stretch",
                help="Inspect Argentina's official CONAE Focos de Calor portal and OGC catalog.",
            )

        # Source-specific input widgets
        uploaded_files = None
        dayfirst_dates = False
        api_days = 5
        api_modis_src = "MODIS_NRT"
        api_viirs_src = "VIIRS_NOAA20_NRT"
        api_start_date = date.today() - timedelta(days=4)

        if source_mode == upload_source_label:
            st.info(
                f"Upload CSV exports from [FIRMS Archive Download]({FIRMS_ARCHIVE_URL}) "
                "(MODIS and/or VIIRS)."
            )
            with st.expander("📥 Step-by-Step: How to Download Multi-Year NASA FIRMS Archives", expanded=False):
                st.markdown(
                    f"""
1. Open [NASA FIRMS Archive Download]({FIRMS_ARCHIVE_URL}).
2. Draw or enter your Bounding Box (or upload the `.geojson` AOI above) and choose date range `2001–Present`.
3. Submit **two separate requests**:
   - **MODIS Standard (`C6.1 / MCD14ML`)** — covers `2001–Present`
   - **VIIRS Suomi NPP (`C2 / VNP14IMGML`)** — covers `2012–Present` (calibration target; keep ≤ `200 MB` per upload)
4. Upload one or both `.csv` files below (max `200 MB` per file on Streamlit Cloud).
                    """
                )
            uploaded_files = st.file_uploader(
                "Select FIRMS CSV files",
                type=["csv"],
                accept_multiple_files=True,
                help="Supports standard FIRMS MODIS (brightness/confidence 0–100) and VIIRS (l/n/h) CSV files.",
            )
            dayfirst_dates = st.checkbox(
                "Parse non-ISO dates as DD/MM/YYYY (dayfirst)",
                value=False,
                help="Enable if Excel/LibreOffice converted ISO YYYY-MM-DD dates into DD/MM/YYYY.",
            )

        elif source_mode == api_source_label:
            st.info(f"Get a free MAP_KEY from [FIRMS Map Key]({FIRMS_MAP_KEY_URL}).")
            server_client = FIRMSApiClient()
            has_server_key = bool(
                server_client.map_key
                and server_client.map_key != "your_firms_map_key_here"
            )
            api_key_input = st.text_input(
                "FIRMS MAP_KEY (optional)",
                value="",
                type="password",
                placeholder=(
                    "Server key active (or paste your own key)"
                    if has_server_key
                    else "Paste your FIRMS MAP_KEY"
                ),
                help="Server key from .env / Streamlit Secrets is used automatically on the backend when left blank.",
            )
            api_client = FIRMSApiClient(map_key=api_key_input or None)

            if (
                st.session_state["api_availability_df"].empty
                and not st.session_state.get("api_availability_attempted", False)
                and api_client.map_key
                and api_client.map_key != "your_firms_map_key_here"
            ):
                st.session_state["api_availability_attempted"] = True
                try:
                    st.session_state["api_availability_df"] = api_client.check_data_availability("ALL")
                except FIRMSApiError:
                    pass

            if st.button("Check API Coverage (Data Availability)", width="stretch"):
                try:
                    avail_df = api_client.check_data_availability("ALL")
                    st.session_state["api_availability_df"] = avail_df
                    st.success(f"Connection verified ({len(avail_df)} sensors available).")
                except FIRMSApiError as exc:
                    st.error(str(exc))

            if not st.session_state["api_availability_df"].empty:
                with st.expander("FIRMS Sensor Availability"):
                    st.dataframe(st.session_state["api_availability_df"], hide_index=True)

            api_modis_src = st.selectbox(
                "MODIS Source",
                ["MODIS_NRT", "MODIS_SP"],
                index=0,
                help="Select MODIS Near Real-Time (NRT) or Standard Processing (SP) archive stream.",
            )
            api_viirs_src = st.selectbox(
                "VIIRS Source",
                ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_SP", "VIIRS_SNPP_SP"],
                index=0,
                help="Select VIIRS NOAA-20 or Suomi NPP NRT/SP stream.",
            )

            per_src_bounds, common_min_dt, common_max_dt = _get_sensor_date_bounds(
                st.session_state["api_availability_df"],
                [api_modis_src, api_viirs_src],
            )
            if per_src_bounds:
                bounds_text = " · ".join(
                    f"`{src}`: {b[0].isoformat()}..{b[1].isoformat()}"
                    for src, b in per_src_bounds.items()
                )
                st.caption(f"Available FIRMS range: {bounds_text}")

            api_days = st.slider(
                "Window Length (days, 5-day chunks)",
                min_value=1,
                max_value=30,
                value=5,
                help="FIRMS Area API serves up to 5 days per call; longer windows are fetched in chunks (max 30 days).",
            )
            api_end_default = common_max_dt or date.today()
            api_start_default = api_end_default - timedelta(days=int(api_days) - 1)
            if common_min_dt and api_start_default < common_min_dt:
                api_start_default = common_min_dt

            api_start_date = st.date_input(
                "Sample Start Date",
                value=api_start_default,
                min_value=common_min_dt,
                max_value=common_max_dt or date.today(),
                key=f"api_start_{api_modis_src}_{api_viirs_src}",
                help="First UTC acquisition date to query from NASA LANCE FIRMS.",
            )

            if st.button("Fetch Data via FIRMS API", type="primary", width="stretch"):
                is_api_bbox_ok, api_bbox_err = validate_bbox(
                    active_bbox,
                    is_live_api=True,
                    max_span_deg=10.0,
                    api_days=int(api_days),
                )
                if not is_api_bbox_ok:
                    st.error(f"❌ {api_bbox_err}")
                else:
                    try:
                        req_end = api_start_date + timedelta(days=int(api_days) - 1)
                        m_max = per_src_bounds.get(api_modis_src, (None, req_end))[1] or req_end
                        v_max = per_src_bounds.get(api_viirs_src, (None, req_end))[1] or req_end
                        m_end = min(req_end, m_max)
                        v_end = min(req_end, v_max)
                        m_start = min(api_start_date, m_end)
                        v_start = min(api_start_date, v_end)

                        with st.spinner("Fetching detections from NASA LANCE FIRMS..."):
                            m_api = api_client.fetch_area_range(
                                source=api_modis_src,
                                bbox=active_bbox,
                                start_date=m_start.strftime("%Y-%m-%d"),
                                end_date=m_end.strftime("%Y-%m-%d"),
                                allow_partial=True,
                            )
                            m_failed = list(api_client.last_failed_windows)
                            v_api = api_client.fetch_area_range(
                                source=api_viirs_src,
                                bbox=active_bbox,
                                start_date=v_start.strftime("%Y-%m-%d"),
                                end_date=v_end.strftime("%Y-%m-%d"),
                                allow_partial=True,
                            )
                            v_failed = list(api_client.last_failed_windows)
                        def _fmt_fw(w: Any) -> str:
                            if isinstance(w, dict):
                                return f"{w.get('source', '')} {w.get('start_date', '')}..{w.get('end_date', '')}: {w.get('error', '')}"
                            return str(w)

                        failed_summary = [_fmt_fw(w) for w in m_failed] + [_fmt_fw(w) for w in v_failed]
                        st.session_state["api_failed_windows"] = failed_summary
                        st.session_state["api_modis_df"] = m_api
                        st.session_state["api_viirs_df"] = v_api
                        st.session_state["api_fetch_meta"] = {
                            "bbox": tuple(round(float(v), 4) for v in active_bbox),
                            "modis_source": api_modis_src,
                            "viirs_source": api_viirs_src,
                            "start_date": min(m_start, v_start).isoformat(),
                            "end_date": max(m_end, v_end).isoformat(),
                            "api_days": int(api_days),
                        }
                        if failed_summary:
                            st.warning(
                                f"Partial fetch completed ({len(failed_summary)} chunk(s) failed): "
                                + "; ".join(failed_summary)
                            )
                        st.success(
                            f"Loaded: MODIS={len(m_api)}, VIIRS={len(v_api)} detections "
                            f"({min(m_start, v_start).isoformat()}..{max(m_end, v_end).isoformat()})."
                        )
                    except FIRMSApiError as exc:
                        st.error(str(exc))

        st.subheader("🔬 Filters & Calibration")
        max_year_bound = max(YEAR_RANGE[1], date.today().year) if is_explore_mode else YEAR_RANGE[1]
        year_min, year_max = st.slider(
            "Year Range",
            min_value=YEAR_RANGE[0],
            max_value=max_year_bound,
            value=(YEAR_RANGE[0], max_year_bound),
            help="Restrict analysis to a subset of years (baseline climatology uses 2001–2025 when available).",
        )
        daynight_option = st.selectbox(
            "Overpass Time (daynight)",
            options=["All", "D", "N"],
            format_func=lambda x: {"All": "All (Day + Night)", "D": "Daytime only (D)", "N": "Nighttime only (N)"}[x],
            help="Filter detections by daytime (D) or nighttime (N) satellite overpass.",
        )
        modis_min_conf = st.slider(
            "Min MODIS Confidence (0–100)",
            min_value=0,
            max_value=100,
            value=DEFAULT_MODIS_MIN_CONFIDENCE,
            step=5,
            help="Minimum FIRMS MODIS detection confidence (%) required to keep a fire pixel.",
        )
        viirs_conf = st.multiselect(
            "Allowed VIIRS Confidence",
            options=["l", "n", "h"],
            default=DEFAULT_VIIRS_CONFIDENCE,
            help="VIIRS confidence categories: l (low), n (nominal), h (high).",
        )
        combo_mode = st.selectbox(
            "Post-2012 Stitching Mode",
            options=["calibrated_viirs", "ensemble"],
            format_func=lambda m: (
                "Calibrated VIIRS × α (seamless extension)"
                if m == "calibrated_viirs"
                else "Ensemble (MODIS + VIIRS × α) / 2"
            ),
            help="Choose how post-2012 series are built: MODIS before 2012-03 + α-scaled VIIRS, or 50/50 ensemble.",
        )
        metric_choice = st.radio(
            "Calendar Metric",
            options=["harmonized_hotspots", "harmonized_frp"],
            format_func=lambda m: (
                "Harmonized Hot Spots (1 km cells)"
                if m == "harmonized_hotspots"
                else "Fire Radiative Power (FRP, MW)"
            ),
            help="Switch all charts and KPIs between harmonized 1 km fire cell counts and radiative power (MW).",
        )
        palette_name = st.selectbox(
            "Color Palette (CVD-Friendly)",
            options=list(COLOR_PALETTES.keys()),
            format_func=lambda p: COLOR_PALETTES[p]["label"],
            help="Perceptually uniform, colorblind-safe colorscales for the heatmap and seasonal plots.",
        )

    # -----------------------------------------------------------------------
    # Validate Bounding Box & Execute Selected Pipeline
    # -----------------------------------------------------------------------
    is_live_api_active = source_mode == api_source_label
    is_bbox_valid, bbox_error_msg = validate_bbox(
        active_bbox,
        is_live_api=is_live_api_active,
        max_span_deg=10.0,
        api_days=int(api_days),
    )
    if not is_bbox_valid:
        st.error(f"❌ Invalid Bounding Box: {bbox_error_msg}")
        st.stop()

    if not viirs_conf:
        st.warning("Select at least one VIIRS confidence level (`l`, `n`, or `h`).")
        st.stop()

    start_date_str = f"{year_min:04d}-01-01"
    end_date_str = f"{year_max:04d}-12-31"
    viirs_conf_tuple = tuple(sorted(viirs_conf))
    in_win_meta: Optional[dict[str, Any]] = None
    ref_result: Optional[HarmonizationResult] = None
    csv_report: Optional[dict[str, Any]] = None

    if not is_explore_mode:
        source_kind = "real_cordoba" if is_real_cordoba_mode else "synthetic"
        export_slug = "cordoba_real" if is_real_cordoba_mode else region_key
        active_region_label = (
            "Córdoba Province, Argentina (Real NASA FIRMS 2001–2025)"
            if is_real_cordoba_mode
            else f"{REGION_PRESETS[region_key]['name']} (Synthetic Benchmark)"
        )
        (
            modis_count,
            viirs_count,
            result,
            ref_result,
            calendar_df,
            climatology_df,
            anomalies_df,
            kpis,
        ) = _run_preset_full_bundle_cached(
            region_key=region_key,
            bbox=active_bbox,
            start_date=start_date_str,
            end_date=end_date_str,
            daynight_filter=daynight_option,
            modis_min_conf=modis_min_conf,
            viirs_allowed_conf=viirs_conf_tuple,
            combination_mode=combo_mode,
            metric_choice=metric_choice,
            source_kind=source_kind,
        )
        if result.monthly_df.empty:
            st.warning(
                "No detections matched the selected region, date range, and confidence filters. "
                "Try expanding the Bounding Box, widening the year range, or lowering confidence thresholds."
            )
            st.stop()
    else:
        loc_display = (
            quick_loc
            if quick_loc != "Custom Coordinates (Manual Input)"
            else f"Custom Global BBox [{active_bbox[0]:.2f}, {active_bbox[1]:.2f}, {active_bbox[2]:.2f}, {active_bbox[3]:.2f}]"
        )
        slug_chars = "".join(c.lower() if c.isalnum() else "_" for c in loc_display)
        export_slug = "_".join(part for part in slug_chars.split("_") if part)[:48] or "custom_bbox"

        if source_mode == upload_source_label:
            if not uploaded_files:
                st.warning("Upload one or more MODIS/VIIRS CSV files in the sidebar to run harmonization.")
                st.stop()
            try:
                raw_modis_df, raw_viirs_df, csv_report = load_csv_dataset_with_report(
                    uploaded_files,
                    dayfirst=dayfirst_dates,
                )
            except ValueError as exc:
                st.error(f"❌ CSV Load Error: {exc}")
                st.stop()
            active_region_label = f"Custom FIRMS CSV — {loc_display}"
            cache_source_key = (
                f"{source_mode}:df={int(dayfirst_dates)}:"
                + ",".join(sorted(f"{f.name}:{f.size}" for f in uploaded_files))
            )
            eff_start_key = start_date_str
            eff_end_key = end_date_str
        else:
            raw_modis_df = st.session_state["api_modis_df"]
            raw_viirs_df = st.session_state["api_viirs_df"]
            fetch_meta = st.session_state.get("api_fetch_meta") or {}
            expected_bbox = tuple(round(float(v), 4) for v in active_bbox)
            if not fetch_meta and raw_modis_df.empty and raw_viirs_df.empty:
                st.info(
                    "Click **Fetch Data via FIRMS API** in the sidebar to load near real-time "
                    "NASA FIRMS detections for the selected Bounding Box."
                )
                st.stop()
            if fetch_meta and fetch_meta.get("bbox") != expected_bbox:
                st.info(
                    "Bounding Box updated since the last API fetch. Click **Fetch Data via FIRMS API** "
                    "in the sidebar to load detections for the new coordinates."
                )
                st.stop()
            active_region_label = f"Live NASA FIRMS API — {loc_display}"
            cache_source_key = f"{source_mode}:{api_modis_src}+{api_viirs_src}"
            eff_start_key = str(fetch_meta.get("start_date", start_date_str))
            eff_end_key = str(fetch_meta.get("end_date", end_date_str))

        modis_filtered = filter_by_bbox_and_dates(
            raw_modis_df,
            bbox=active_bbox,
            start_date=start_date_str,
            end_date=end_date_str,
            daynight=daynight_option,
        )
        viirs_filtered = filter_by_bbox_and_dates(
            raw_viirs_df,
            bbox=active_bbox,
            start_date=start_date_str,
            end_date=end_date_str,
            daynight=daynight_option,
        )
        modis_count = len(modis_filtered)
        viirs_count = len(viirs_filtered)

        all_dates = pd.concat(
            [
                modis_filtered["acq_date"] if not modis_filtered.empty else pd.Series(dtype=str),
                viirs_filtered["acq_date"] if not viirs_filtered.empty else pd.Series(dtype=str),
            ],
            ignore_index=True,
        )
        unique_years_count = int(all_dates.astype(str).str[:4].nunique()) if not all_dates.empty else 0

        cal_override: Optional[pd.DataFrame] = None
        if unique_years_count < 2:
            cal_override, in_win_meta = _compute_in_window_calibration_cached(
                modis_df=modis_filtered,
                viirs_df=viirs_filtered,
                modis_min_conf=modis_min_conf,
                viirs_allowed_conf=viirs_conf_tuple,
                daynight_filter=daynight_option,
                bbox=active_bbox,
                source_key=cache_source_key,
                start_date=eff_start_key,
                end_date=eff_end_key,
            )

        result = _run_harmonization_cached(
            modis_df=modis_filtered,
            viirs_df=viirs_filtered,
            modis_min_conf=modis_min_conf,
            viirs_allowed_conf=viirs_conf_tuple,
            combination_mode=combo_mode,
            daynight_filter=daynight_option,
            calibration_override=cal_override,
            bbox=active_bbox,
            source_key=cache_source_key,
            start_date=eff_start_key,
            end_date=eff_end_key,
        )

        if result.monthly_df.empty:
            st.warning(
                "No detections matched the selected region, date range, and confidence filters. "
                "Try expanding the Bounding Box, widening the year range, or lowering confidence thresholds."
            )
            st.stop()

        calendar_df, climatology_df, anomalies_df, kpis = _compute_climatology_bundle_cached(
            monthly_df=result.monthly_df,
            gridded_df=result.gridded_df,
            transition_metrics=result.transition_metrics,
            ref_monthly_df=None,
            metric_choice=metric_choice,
            bbox=active_bbox,
            source_key=cache_source_key,
            start_date=eff_start_key,
            end_date=eff_end_key,
        )

    use_frp = metric_choice == "harmonized_frp"
    metric_label = "Harmonized FRP (MW)" if use_frp else "Harmonized Hot Spots (1 km cells)"
    tm = result.transition_metrics
    is_in_window_explore = is_explore_mode and (in_win_meta is not None)
    is_ref_mode = (not is_explore_mode) and bool(tm.get("used_reference_calibration", False))
    has_overlap = bool(
        is_ref_mode
        or (
            modis_count > 0
            and viirs_count > 0
            and not bool(result.calibration_df["used_fallback"].all())
            and bool(tm.get("has_valid_overlap", True))
        )
    )

    area_km2 = _compute_bbox_area_km2(active_bbox)
    st.markdown(
        f"**Region:** {active_region_label} · **BBox Area:** {area_km2:,.0f} km² · "
        f"**Filtered Detections:** MODIS `{modis_count:,}` | VIIRS `{viirs_count:,}`"
    )
    if is_real_cordoba_mode:
        st.success(
            "✅ Real NASA FIRMS Satellite Archive: MODIS C6.1 (MCD14DL) + Suomi NPP VIIRS C2 (VNP14IMG) "
            "· Córdoba Province, Argentina (2001–2025)."
        )
    elif (not is_explore_mode) and source_mode == synth_source_label:
        st.warning(
            "🧪 **DEMO / Benchmark (Synthetic Validation 2001–2026):** "
            "Algorithmic self-consistency benchmark (`data/sample/`) with known latent fire activity used to verify "
            "2012 step-jump suppression and out-of-sample α recovery. For real satellite observations, "
            "use **Real NASA FIRMS Archive — Córdoba (2001–2025)** or switch to **Explore Any Region**."
        )
    if (not is_in_window_explore) and (not has_overlap):
        st.warning(
            "⚠️ **Uncalibrated Series (α = 1.0 fallback):** "
            "No paired MODIS–VIIRS overlap was found in the selected window/archive, so cross-sensor scaling α could not be calibrated."
        )
    if is_live_api_active and st.session_state.get("api_failed_windows"):
        failed_list = st.session_state["api_failed_windows"]
        st.warning(
            f"⚠️ **Partial Live API Response ({len(failed_list)} 5-day chunk(s) failed):** "
            + "; ".join(failed_list)
        )
    if csv_report is not None:
        total_mb = float(csv_report.get("total_size_bytes", 0)) / (1024.0 * 1024.0)
        data_bb = csv_report.get("data_bbox")
        disjoint_bbox = False
        if isinstance(data_bb, (tuple, list)) and len(data_bb) == 4:
            dw, ds, de, dn = data_bb
            aw, as_, ae, an = active_bbox
            disjoint_bbox = de < aw or dw > ae or dn < as_ or ds > an
        if disjoint_bbox:
            st.warning(
                f"⚠️ Uploaded CSV extent `[{data_bb[0]:.2f}, {data_bb[1]:.2f}, {data_bb[2]:.2f}, {data_bb[3]:.2f}]` "
                f"does not overlap active BBox `[{active_bbox[0]:.2f}, {active_bbox[1]:.2f}, {active_bbox[2]:.2f}, {active_bbox[3]:.2f}]`."
            )
        if total_mb >= 180.0:
            st.warning(f"⚠️ Uploaded CSV size ({total_mb:.1f} MB) is near Streamlit's 200 MB per-upload limit.")
        with st.expander("📋 CSV Load Diagnostics", expanded=False):
            st.markdown(
                f"- **Rows Read:** `{csv_report['rows_read']:,}` → **Kept:** `{csv_report['rows_kept']:,}` "
                f"(dropped coords: `{csv_report['rows_dropped_coords']:,}`, invalid dates: `{csv_report['rows_dropped_dates']:,}`)\n"
                f"- **Detected Range:** `{csv_report['date_min']} .. {csv_report['date_max']}` · **Upload Size:** `{total_mb:.2f} MB` (limit `200 MB`)\n"
                f"- **Delimiters Detected:** `{', '.join(csv_report['delimiters'])}`"
            )
    if is_ref_mode:
        cal_file_label = (
            "calibration_alpha_cordoba_real.csv"
            if (is_real_cordoba_mode or region_key == "cordoba_argentina")
            else f"calibration_alpha_{region_key}.csv"
        )
        st.info(
            f"ℹ️ **Short-Window Mode (< 2 years):** "
            f"applying pre-computed regional calibration factors `α(m, daynight)` from "
            f"`data/processed/{cal_file_label}` and the 2001–2025 baseline climatology "
            f"(active window: `{tm.get('live_raw_points', 0):.0f}` raw detections → "
            f"`{tm.get('live_gridded_cells', 0):.0f}` 0.01° cells → "
            f"`{tm.get('live_harmonized_hotspots', 0):.1f}` harmonized hotspots)."
        )

    daily_breakdown_df = pd.DataFrame()
    if is_in_window_explore and in_win_meta is not None:
        n_m_cells = int(in_win_meta["n_modis_cells"])
        n_v_cells = int(in_win_meta["n_viirs_cells"])
        st.info(
            f"🌐 **Custom Global Window ({eff_start_key} .. {eff_end_key}):** "
            f"Active BBox `[{active_bbox[0]:.3f}, {active_bbox[1]:.3f}, {active_bbox[2]:.3f}, {active_bbox[3]:.3f}]` "
            f"· Area `{area_km2:,.0f} km²` · `alpha from {n_m_cells} MODIS / {n_v_cells} VIIRS 0.01° cells` "
            f"({in_win_meta['mode_label']})."
        )
        if in_win_meta["low_sample_warning"]:
            st.warning(
                f"⚠️ **Below 30 Active 0.01° Cells Threshold:** "
                f"Detected `{n_m_cells}` MODIS and `{n_v_cells}` VIIRS 0.01° cells (< {in_win_meta['min_cells_threshold']} cells). "
                "Using `alpha = 1.0` (`VIIRS native, not harmonized to MODIS`) rather than cross-sensor scaling."
            )
        if not in_win_meta["used_fallback"]:
            st.caption(
                "Note: Harmonized count equals the MODIS-equivalent count by construction in this in-window mode."
            )
        daily_breakdown_df = build_daily_sensor_breakdown(result.gridded_df, result.harmonized_cells_df)

    # -----------------------------------------------------------------------
    # Top KPI Row
    # -----------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    if is_in_window_explore and in_win_meta is not None:
        if not daily_breakdown_df.empty:
            rank_col = "harmonized_frp" if use_frp else "harmonized_cells"
            peak_idx = int(daily_breakdown_df[rank_col].idxmax())
            peak_row = daily_breakdown_df.loc[peak_idx]
            peak_date_str = str(peak_row["acq_date"])
            peak_day_cells = float(peak_row["harmonized_cells"])
            peak_day_frp = float(peak_row["harmonized_frp"])
        else:
            peak_date_str = "—"
            peak_day_cells = 0.0
            peak_day_frp = 0.0

        n_modis_cells = int(in_win_meta["n_modis_cells"])
        n_viirs_cells = int(in_win_meta["n_viirs_cells"])
        live_harmonized = (
            float(daily_breakdown_df["harmonized_cells"].sum())
            if not daily_breakdown_df.empty
            else float(tm.get("live_harmonized_hotspots", 0.0))
        )
        live_raw = float(tm.get("live_raw_points", 0.0))
        modis_frp_total = float(in_win_meta["modis_frp_total"])
        viirs_frp_total = float(in_win_meta["viirs_frp_total"])
        harm_frp_total = (
            float(daily_breakdown_df["harmonized_frp"].sum())
            if not daily_breakdown_df.empty
            else float(result.monthly_df["harmonized_frp"].sum())
        )

        k1.metric(
            "Peak Activity Date",
            peak_date_str,
            delta=f"{peak_day_cells:.1f} cells · {peak_day_frp:,.1f} MW",
            delta_color="off",
            help="UTC acquisition date in the selected window with peak fire activity.",
        )
        k2.metric(
            "Window 0.01° Cells",
            f"{live_harmonized:.1f}",
            delta=f"MODIS: {n_modis_cells} · VIIRS: {n_viirs_cells} (Raw pts: {live_raw:.0f})",
            delta_color="off",
            help="Harmonized 0.01° fire cells alongside native MODIS/VIIRS cell counts and raw detections.",
        )
        k3.metric(
            "Sensor FRP (Separate)",
            f"VIIRS: {viirs_frp_total:,.1f} MW",
            delta=f"MODIS: {modis_frp_total:,.1f} MW · Harm: {harm_frp_total:,.1f} MW",
            delta_color="off",
            help="Separate radiative power totals per sensor (MODIS and VIIRS FRP are never summed directly).",
        )
        if in_win_meta["used_fallback"]:
            k4.metric(
                "Window Calibration α",
                "1.000 (Native)",
                delta="VIIRS native (not harmonized)",
                delta_color="off",
                help="Sample size below 30 active 0.01° cells per sensor; reporting native unscaled VIIRS detections.",
            )
        else:
            applied_alpha = float(in_win_meta["overall_alpha_cells"])
            k4.metric(
                "Window Calibration α",
                f"{applied_alpha:.3f}",
                delta=f"From {n_modis_cells} MODIS / {n_viirs_cells} VIIRS cells",
                delta_color="off",
                help="In-window empirical MODIS-equivalent scaling ratio computed strictly from active cells in this window.",
            )
    else:
        k1.metric(
            "Peak Fire Month",
            kpis.get("peak_month_name", "—"),
            delta=f"Median: {kpis.get('peak_month_median', 0):,.1f}",
            delta_color="off",
            help="Calendar month with the highest historical median fire activity in the dataset.",
        )
        if is_ref_mode:
            k2.metric(
                "Window Hot Spots (< 2 yrs)",
                f"{tm.get('live_harmonized_hotspots', 0):.1f}",
                delta=f"Raw: {tm.get('live_raw_points', 0):.0f} · Cells: {tm.get('live_gridded_cells', 0):.0f}",
                delta_color="off",
                help="Total α-calibrated 1 km fire hotspots detected within the active short window.",
            )
        else:
            worst_frp = float(kpis.get("worst_year_frp", 0.0))
            frp_text = f"{worst_frp / 1000.0:.1f} GW" if worst_frp >= 1000.0 else f"{worst_frp:,.0f} MW"
            k2.metric(
                "Peak Wildfire Year",
                str(kpis.get("worst_year", "—")),
                delta=f"{kpis.get('worst_year_hotspots', 0):,.0f} cells · {frp_text}",
                delta_color="off",
                help="Year with the highest cumulative harmonized fire activity (hotspots & total FRP) in the selected range.",
            )
        k3.metric(
            f"Season {kpis.get('focus_year', '')} ({kpis.get('focus_month_name', '')})",
            f"{kpis.get('focus_anomaly_pct', 0):+.1f}%",
            delta=f"{kpis.get('focus_robust_z_score', kpis.get('focus_z_score', 0)):+.2f} R-Z · {_compact_status(kpis.get('focus_status', ''))}",
            delta_color="inverse",
            help="Percentage departure and Leave-One-Out Robust Z-score (0.7413·IQR) vs. the monthly climatological baseline.",
        )
        if has_overlap:
            k4.metric(
                "Calibration α (VIIRS)",
                f"{tm.get('mean_alpha_overall', 1.0):.3f}",
                delta=f"Overlap Bias: {tm.get('overlap_sensor_residual_bias_pct', 0.0):+.2f}% · D:{tm.get('mean_alpha_day', 1.0):.2f} N:{tm.get('mean_alpha_night', 1.0):.2f}",
                delta_color="off",
                help="Sum-ratio VIIRS (375 m) → MODIS (1 km) scaling factor across daytime (D) and nighttime (N) overpasses.",
            )
        else:
            k4.metric(
                "Calibration α (VIIRS)",
                "N/A",
                delta="No MODIS–VIIRS overlap in window",
                delta_color="off",
                help="Cross-sensor calibration requires paired MODIS and VIIRS observations (2012–2025).",
            )

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        [
            "📅 Burning Activity Calendar",
            "📊 Harmonization Proof (Raw vs. Harmonized)",
            "🗺️ Geospatial Explorer & CSV Export",
        ]
    )

    # === TAB 1: BURNING ACTIVITY CALENDAR ===
    with tab1:
        if is_in_window_explore:
            st.subheader("📅 Operational Window — Daily Fire Activity")
            st.plotly_chart(
                _make_live_daily_chart(
                    gridded_df=result.gridded_df,
                    use_frp=use_frp,
                    harmonized_cells_df=result.harmonized_cells_df,
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
            )
            st.subheader("Daily Per-Sensor Counts & Separate FRP Breakdown")
            _render_daily_breakdown_table(daily_breakdown_df)
        else:
            with st.expander("ℹ️ How to Read This Calendar", expanded=False):
                st.markdown(
                    """
- **Hotspot Cell ≠ Fire Perimeter:** Each unit is a unique active `0.01° × 0.01°` (`~1 km`) grid cell per day and Day/Night overpass (`satellite + overpass`), not individual burned-area scars.
- **Sum-Ratio Calibration `α(D/N, m)`:** Post-2012 VIIRS (`375 m`) counts are scaled toward MODIS (`1 km`) equivalents using per-overpass sum ratios from the `2012–2025` overlap with paired-day bootstrap 95% CIs.
- **Gray Cell = `No Data`:** Unobserved pre-launch/gap months remain blank gray (`NaN`) rather than `0`, and are excluded from climatology medians and percentiles.
- **Leave-One-Out Robust Z-Score:** Each month's anomaly is evaluated against all other observed baseline years (`year != target`) using `Robust Z = (value - median) / max(0.7413 · IQR, 1)`.
                    """
                )

            nonstat = check_series_nonstationarity(result.monthly_df, value_col=metric_choice)
            if nonstat.get("has_significant_trend"):
                st.warning(f"📉 **Non-Stationary Baseline:** {nonstat['message']}")

            color_scale_mode = st.radio(
                "Color Scale Mode",
                options=COLOR_SCALE_MODES,
                index=0,
                horizontal=True,
                help=(
                    "Robust (P95 Clamped) caps the heatmap color limit at the 95th percentile of active months "
                    "so 1–2 extreme mega-fire seasons do not wash out normal seasonal patterns. "
                    "Cell text and hover tooltips always show exact real values; gray cells indicate No Data."
                ),
            )
            st.plotly_chart(
                _make_calendar_heatmap(
                    calendar_df=calendar_df,
                    colorscale=COLOR_PALETTES[palette_name]["plotly"],
                    metric_label=metric_label,
                    color_scale_mode=color_scale_mode,
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
            )

            available_years = sorted(anomalies_df["year"].unique().tolist(), reverse=True)
            default_cmp_year = kpis.get("worst_year", available_years[0])
            if default_cmp_year not in available_years:
                default_cmp_year = available_years[0]

            c_left, c_right = st.columns([1, 2])
            with c_left:
                selected_year = st.selectbox(
                    "Comparison Year vs. Baseline",
                    options=available_years,
                    index=available_years.index(default_cmp_year),
                    help="Select a year to plot against the historical median and P25–P75 interquartile band.",
                )
            with c_right:
                anom_metric_mode = st.radio(
                    "Anomaly Metric",
                    options=["Robust Z-score (0.7413·IQR)", "% Deviation from Median", "Z-score (σ)"],
                    index=0,
                    horizontal=True,
                    help="View monthly departures via Leave-One-Out Robust Z-score (0.7413·IQR), percentage from median, or classical σ.",
                )

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.plotly_chart(
                    _make_seasonal_profile_chart(
                        climatology_df=climatology_df,
                        anomalies_df=anomalies_df,
                        target_year=selected_year,
                        metric_label=metric_label,
                        accent_color=COLOR_PALETTES[palette_name]["accent"],
                    ),
                    width="stretch",
                    config=PLOTLY_CONFIG,
                )
            with p_col2:
                st.plotly_chart(
                    _make_anomaly_bar_chart(
                        anomalies_df=anomalies_df,
                        target_year=selected_year,
                        mode=anom_metric_mode,
                    ),
                    width="stretch",
                    config=PLOTLY_CONFIG,
                )

            with st.expander(f"Detailed Monthly Anomaly Table for {selected_year}"):
                year_cols = [
                    "date_label",
                    "month_name",
                    "value",
                    "median",
                    "p25",
                    "p75",
                    "anomaly_abs",
                    "anomaly_pct",
                    "robust_z_score",
                    "z_score",
                    "anomaly_status",
                    "primary_source",
                ]
                avail_cols = [c for c in year_cols if c in anomalies_df.columns]
                year_table = anomalies_df[anomalies_df["year"] == selected_year][avail_cols].rename(
                    columns={
                        "date_label": "Period",
                        "month_name": "Month",
                        "value": "Harmonized",
                        "median": "LOO Median",
                        "p25": "LOO P25",
                        "p75": "LOO P75",
                        "anomaly_abs": "Deviation (Abs.)",
                        "anomaly_pct": "Deviation (%)",
                        "robust_z_score": "Robust Z",
                        "z_score": "Z-score",
                        "anomaly_status": "Status",
                        "primary_source": "Series Source",
                    }
                )
                st.dataframe(year_table, hide_index=True, width="stretch")

            tab1_daily_src = (
                ref_result if (is_ref_mode and ref_result is not None) else result
            )
            tab1_daily = build_daily_sensor_breakdown(
                tab1_daily_src.gridded_df,
                tab1_daily_src.harmonized_cells_df,
            )
            crit_df = compute_critical_fire_windows(
                tab1_daily,
                value_col="harmonized_frp" if use_frp else "harmonized_cells",
            )
            if not crit_df.empty:
                st.subheader("🔥 Critical Fire Risk Windows (52-Week Exceedance P > P75)")
                crit_weeks = crit_df[crit_df["is_critical"]]
                top_crit = crit_df.sort_values(["p_exceedance", "mean_weekly"], ascending=False).iloc[0]
                if not crit_weeks.empty:
                    crit_span = f"{crit_weeks.iloc[0]['week_label']} – {crit_weeks.iloc[-1]['week_label']} ({len(crit_weeks)} critical weeks >= 50%)"
                else:
                    crit_span = f"Peak week {top_crit['week_label']} ({top_crit['p_exceedance'] * 100:.1f}% > P75)"
                st.caption(
                    f"Peak Risk Window: **{top_crit['week_label']}** (`{top_crit['p_exceedance'] * 100:.1f}%` of years above P75 = `{top_crit['threshold_p75']:,.1f}`) · Active Window: `{crit_span}`"
                )
                st.plotly_chart(
                    _make_critical_windows_chart(crit_df),
                    width="stretch",
                    config=PLOTLY_CONFIG,
                )

    # === TAB 2: HARMONIZATION PROOF ===
    with tab2:
        if is_in_window_explore and in_win_meta is not None:
            st.subheader("📊 In-Window Sensor Comparison & Honest Calibration")
            st.plotly_chart(
                _make_live_daily_chart(
                    gridded_df=result.gridded_df,
                    use_frp=use_frp,
                    harmonized_cells_df=result.harmonized_cells_df,
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Active MODIS 0.01° Cells",
                f"{int(in_win_meta['n_modis_cells']):,}",
                delta=f"FRP: {in_win_meta['modis_frp_total']:,.1f} MW",
                delta_color="off",
            )
            c2.metric(
                "Active VIIRS 0.01° Cells",
                f"{int(in_win_meta['n_viirs_cells']):,}",
                delta=f"FRP: {in_win_meta['viirs_frp_total']:,.1f} MW",
                delta_color="off",
            )
            c3.metric(
                "Day / Night α Ratio",
                f"D:{in_win_meta['alpha_day']:.3f} / N:{in_win_meta['alpha_night']:.3f}",
                delta=f"Overall: {in_win_meta['overall_alpha_cells']:.3f}",
                delta_color="off",
            )
            c4.metric(
                "Calibration Regime",
                "Native (α=1.0)" if in_win_meta["used_fallback"] else "In-Window Emp.",
                delta=in_win_meta["mode_label"],
                delta_color="off",
            )
            st.subheader("Daily Per-Sensor Breakdown Table")
            _render_daily_breakdown_table(daily_breakdown_df)
            with st.expander("In-Window Calibration Table α(month, daynight)"):
                st.dataframe(result.calibration_df, hide_index=True, width="stretch")
        else:
            if is_ref_mode and ref_result is not None:
                st.plotly_chart(
                    _make_live_daily_chart(
                        gridded_df=result.gridded_df,
                        use_frp=use_frp,
                        harmonized_cells_df=result.harmonized_cells_df,
                    ),
                    width="stretch",
                    config=PLOTLY_CONFIG,
                )
                proof_monthly_df = ref_result.monthly_df
            else:
                proof_monthly_df = result.monthly_df

            granularity = st.radio(
                "Time Series Granularity",
                options=["Annual Total", "Monthly Series"],
                horizontal=True,
                help="Toggle between annual sums and full monthly resolution across the 2012 VIIRS transition.",
            )
            st.plotly_chart(
                _make_raw_vs_harmonized_chart(
                    monthly_df=proof_monthly_df,
                    granularity=granularity,
                    use_frp=use_frp,
                ),
                width="stretch",
                config=PLOTLY_CONFIG,
            )

            if is_ref_mode:
                cal_csv_name = (
                    "calibration_alpha_cordoba_real.csv"
                    if (is_real_cordoba_mode or region_key == "cordoba_argentina")
                    else f"calibration_alpha_{region_key}.csv"
                )
                st.caption(
                    f"2012 transition diagnostics and overlap R² below reflect the reference archive for "
                    f"`{REGION_PRESETS[region_key]['name']}` (2001–2025 complete years), as the active window spans < 2 years "
                    f"and uses pre-calibrated `{cal_csv_name}`."
                )

            if not has_overlap:
                st.warning(
                    "⚠️ **No MODIS–VIIRS overlap in the selected window:** "
                    "Cross-sensor 2012 step-jump suppression, residual overlap bias, and regression metrics "
                    "require paired MODIS and VIIRS observations (2012–2025). Expand the Year Range or include "
                    "both sensors to evaluate transition metrics."
                )

            m1, m2, m3, m4 = st.columns(4)
            if has_overlap:
                pre_s = int(tm.get("pre2012_start_year", 2003))
                m1.metric(
                    "In-Sample Overlap Bias (2012–25)",
                    f"{tm.get('overlap_sensor_residual_bias_pct', 0):+.2f}%",
                    delta=f"Out-of-Sample Holdout (2019–25): {tm.get('holdout_bias_pct', 0):+.2f}%",
                    delta_color="off",
                    help=(
                        "Sum-ratio residual bias over concurrent MODIS–VIIRS overlap (2012-03..2025) and "
                        "out-of-sample holdout test (α trained on 2012–2018, evaluated on 2019–2025)."
                    ),
                )
                m2.metric(
                    f"Post-2012 Shift ({pre_s}–11 vs 13–25)",
                    f"{tm.get('harmonized_diff_pct', 0):+.1f}% (Harm.)",
                    delta=f"MODIS Natural: {tm.get('modis_natural_change_pct', 0):+.1f}% · Raw jump: {tm.get('raw_combined_jump_pct', 0):+.1f}%",
                    delta_color="off",
                    help=(
                        "Harmonized change vs. consistent MODIS-only physical trend over complete years "
                        f"({pre_s}–2011 vs. 2013–2025), alongside the unharmonized raw point step-jump."
                    ),
                )
                m3.metric(
                    "Overlap Agreement (2012–25)",
                    f"R² = {tm.get('overlap_r2', 0):.4f}",
                    delta=f"Deseas. Anomaly r = {tm.get('deseasonalized_anomaly_r', 0):.4f} · r = {tm.get('overlap_pearson_r', 0):.4f}",
                    delta_color="off",
                    help=(
                        "Monthly variance explained (R²), Pearson r, and deseasonalized anomaly correlation r "
                        "between MODIS and α-calibrated VIIRS across overlap months."
                    ),
                )
                m4.metric(
                    "Sum-Ratio α (VIIRS → MODIS)",
                    f"{tm.get('mean_alpha_overall', 1.0):.3f}",
                    delta=f"95% CI Width: ±{tm.get('alpha_ci_width_mean', 0):.3f} · D:{tm.get('mean_alpha_day', 1.0):.2f} N:{tm.get('mean_alpha_night', 1.0):.2f}",
                    delta_color="off",
                    help=(
                        "Overall sum-ratio VIIRS → MODIS scaling factor, mean 95% paired bootstrap CI width, "
                        "and daytime (D) / nighttime (N) calibration weights."
                    ),
                )
            else:
                for col_card, title_str in [
                    (m1, "In-Sample Overlap Bias (2012–25)"),
                    (m2, "Post-2012 Shift"),
                    (m3, "Overlap Agreement (2012–25)"),
                    (m4, "Sum-Ratio α (VIIRS → MODIS)"),
                ]:
                    col_card.metric(
                        title_str,
                        "N/A",
                        delta="No MODIS–VIIRS overlap in window",
                        delta_color="off",
                    )

            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.plotly_chart(
                    _make_overlap_scatter(result.overlap_monthly_df, tm),
                    width="stretch",
                    config=PLOTLY_CONFIG,
                )
            with s_col2:
                st.plotly_chart(
                    _make_alpha_bar_chart(result.calibration_df),
                    width="stretch",
                    config=PLOTLY_CONFIG,
                )

            with st.expander("Monthly & Day/Night Calibration Table α(m, daynight) & 95% Bootstrap CI"):
                st.dataframe(result.calibration_df, hide_index=True, width="stretch")

    # === TAB 3: GEOSPATIAL EXPLORER & CSV EXPORT ===
    with tab3:
        st.markdown(
            f"Inspect harmonized 0.01° cells alongside external OGC WMS overlays "
            f"([NASA FIRMS]({FIRMS_PORTAL_URL}) / [Global Fire Map]({FIRMS_MAP_URL})"
            + (
                f", [CONAE OGC]({CONAE_CATALOG_OGC_URL}), [CONAE Focos]({CONAE_FOCOS_URL}))."
                if not is_explore_mode
                else ")."
            )
        )
        if not is_explore_mode:
            b1, b2 = st.columns(2)
            b1.link_button(
                "🌐 Compare in Official NASA FIRMS Fire Map",
                firms_compare_url,
                width="stretch",
                help="Open this exact region and zoom level on NASA FIRMS Official Fire Map.",
            )
            b2.link_button(
                "🛰️ Open Official CONAE Geoportal",
                CONAE_FOCOS_URL,
                width="stretch",
                help="Open Argentina's CONAE Focos de Calor active-fire viewer.",
            )
        else:
            st.link_button(
                "🌐 Compare in Official NASA FIRMS Fire Map",
                firms_compare_url,
                width="stretch",
                help="Open this exact region and zoom level on NASA FIRMS Official Fire Map.",
            )

        f1, f2, f3 = st.columns(3)
        map_years = ["All Years"] + sorted(result.gridded_df["year"].unique().tolist(), reverse=True)
        default_map_year = None if is_in_window_explore else kpis.get("worst_year")
        default_map_idx = (
            map_years.index(default_map_year)
            if default_map_year in map_years
            else (1 if len(map_years) > 1 and not is_in_window_explore else 0)
        )
        map_year_sel = f1.selectbox(
            "Map Year",
            options=map_years,
            index=default_map_idx,
            key=f"map_year_{export_slug}_{source_mode}",
            help="Filter displayed grid cells and thermal heatmap to a specific year or all years.",
        )
        map_month_sel = f2.selectbox(
            "Map Month",
            options=["All Months"] + list(range(1, 13)),
            format_func=lambda m: m if isinstance(m, str) else f"{m:02d} — {MONTH_LABELS[m - 1]}",
            help="Filter displayed grid cells and thermal heatmap to a specific calendar month.",
        )
        max_markers = f3.slider(
            "Max Vector Markers",
            min_value=100,
            max_value=1500,
            value=300,
            step=100,
            help="Maximum number of top-FRP circle markers rendered individually (HeatMap always renders all cells).",
        )

        map_df = result.gridded_df.copy()
        if map_year_sel != "All Years":
            map_df = map_df[map_df["year"] == int(map_year_sel)]
        if map_month_sel != "All Months":
            map_df = map_df[map_df["month"] == int(map_month_sel)]

        fmap = _build_folium_map(map_df, active_bbox, max_points=max_markers)
        map_html = fmap.get_root().render()
        if hasattr(st, "iframe"):
            st.iframe(map_html, height=540, width="stretch")
        else:
            import streamlit.components.v1 as components

            components.html(map_html, height=540, scrolling=False)

        st.divider()
        st.subheader("📥 Export Harmonized Datasets (CSV)")
        d1, d2 = st.columns(2)
        d3, d4 = st.columns(2)

        d1.download_button(
            label="harmonized_hotspots.csv",
            data=result.harmonized_cells_df.to_csv(index=False).encode("utf-8"),
            file_name=f"harmonized_hotspots_{export_slug}.csv",
            mime="text/csv",
            width="stretch",
        )
        if is_in_window_explore:
            d2.download_button(
                label="daily_sensor_breakdown.csv",
                data=daily_breakdown_df.to_csv(index=False).encode("utf-8"),
                file_name=f"daily_sensor_breakdown_{export_slug}.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            d2.download_button(
                label="burning_activity_calendar.csv",
                data=calendar_df.to_csv().encode("utf-8"),
                file_name=f"burning_activity_calendar_{export_slug}.csv",
                mime="text/csv",
                width="stretch",
            )
        d3.download_button(
            label="harmonized_monthly_series.csv",
            data=anomalies_df.to_csv(index=False).encode("utf-8"),
            file_name=f"harmonized_monthly_series_{export_slug}.csv",
            mime="text/csv",
            width="stretch",
        )
        d4.download_button(
            label="calibration_alpha_table.csv",
            data=result.calibration_df.to_csv(index=False).encode("utf-8"),
            file_name=f"calibration_alpha_{export_slug}.csv",
            mime="text/csv",
            width="stretch",
        )

        manifest_payload = {
            "code_version": "1.0.0",
            "operating_mode": operating_mode,
            "source_mode": source_mode,
            "region_label": active_region_label,
            "bbox": [float(v) for v in active_bbox],
            "bbox_area_km2": round(float(area_km2), 2),
            "year_range": [int(year_min), int(year_max)],
            "daynight_filter": daynight_option,
            "modis_min_conf": int(modis_min_conf),
            "viirs_allowed_conf": list(viirs_conf_tuple),
            "combination_mode": combo_mode,
            "metric_choice": metric_choice,
            "filtered_counts": {"modis": int(modis_count), "viirs": int(viirs_count)},
            "transition_metrics": {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in tm.items()},
        }
        st.download_button(
            label="📄 run_metadata_manifest.json",
            data=json.dumps(manifest_payload, indent=2).encode("utf-8"),
            file_name=f"run_metadata_manifest_{export_slug}.json",
            mime="application/json",
            width="stretch",
        )


if __name__ == "__main__":
    main()
