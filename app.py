"""Interactive Streamlit dashboard for MODIS & VIIRS Hot Spots Harmonization.

NASA Space Apps Challenge 2026 — Harmonization of MODIS and VIIRS Hot Spots (NASA & CONAE).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

import folium
from folium.plugins import Draw, HeatMap
from folium.template import Template
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyproj import Geod
import streamlit as st

from src.analytics import (
    MONTH_LABELS,
    build_calendar_matrix,
    compute_kpi_metrics,
    compute_monthly_anomalies,
    compute_monthly_climatology,
)
from src.config import (
    COLOR_PALETTES,
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
    REGION_PRESETS,
    YEAR_RANGE,
)
from src.data_loader import (
    FIRMSApiClient,
    FIRMSApiError,
    filter_by_bbox_and_dates,
    generate_regional_sample_data,
    load_csv_dataset,
)
from src.harmonizer import HarmonizationPipeline, HarmonizationResult

st.set_page_config(
    page_title="MODIS & VIIRS Fire Harmonizer | NASA Space Apps 2026",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Cached Data & Pipeline Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Расчёт гармонизации MODIS и VIIRS...")
def _load_sample_cached(region_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_regional_sample_data(region_key=region_key, force=False)


@st.cache_data(show_spinner="Расчёт гармонизации MODIS и VIIRS...")
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
) -> HarmonizationResult:
    pipeline = HarmonizationPipeline(
        modis_min_conf=modis_min_conf,
        viirs_allowed_conf=list(viirs_allowed_conf),
        combination_mode=combination_mode,
        daynight_filter=daynight_filter,
    )
    return pipeline.run(
        modis_df,
        viirs_df,
        external_calibration_df=external_calibration_df,
        external_overlap_df=external_overlap_df,
        reference_transition_metrics=reference_transition_metrics,
    )


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner="Расчёт гармонизации MODIS и VIIRS...")
def _load_regional_reference_cached(
    region_key: str,
    modis_min_conf: int,
    viirs_allowed_conf: tuple[str, ...],
    combination_mode: str,
    daynight_filter: str,
) -> tuple[HarmonizationResult, pd.DataFrame]:
    """Load pre-computed calibration & full 2001–2026 reference baseline for short windows."""
    ref_modis, ref_viirs = _load_sample_cached(region_key)
    ref_modis_dn = filter_by_bbox_and_dates(ref_modis, daynight=daynight_filter)
    ref_viirs_dn = filter_by_bbox_and_dates(ref_viirs, daynight=daynight_filter)

    ref_result = _run_harmonization_cached(
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
    cal_df = _load_regional_calibration_csv(region_key) if use_default_csv else None
    if cal_df is None or cal_df.empty:
        cal_df = ref_result.calibration_df.copy()

    return ref_result, cal_df


@st.cache_data(show_spinner="Расчёт гармонизации MODIS и VIIRS...")
def _compute_climatology_bundle_cached(
    monthly_df: pd.DataFrame,
    gridded_df: pd.DataFrame,
    transition_metrics: dict[str, Any],
    ref_monthly_df: Optional[pd.DataFrame],
    metric_choice: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute and cache calendar matrix, seasonal climatology, anomalies, and KPIs."""
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


@st.cache_data(show_spinner="Расчёт гармонизации MODIS и VIIRS...")
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
    """End-to-end cached preset loading, filtering, harmonization, and climatology."""
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
        )
        ext_overlap_df = ref_result.overlap_monthly_df
        ref_tm = ref_result.transition_metrics

    result = _run_harmonization_cached(
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
    if result.monthly_df.empty:
        empty_df = pd.DataFrame()
        return len(modis_filtered), len(viirs_filtered), result, ref_result, empty_df, empty_df, empty_df, {}

    calendar_df, climatology_df, anomalies_df, kpis = _compute_climatology_bundle_cached(
        monthly_df=result.monthly_df,
        gridded_df=result.gridded_df,
        transition_metrics=result.transition_metrics,
        ref_monthly_df=ref_result.monthly_df if ref_result is not None else None,
        metric_choice=metric_choice,
    )
    return (
        len(modis_filtered),
        len(viirs_filtered),
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


def _apply_bbox_to_state(bbox: tuple[float, float, float, float]) -> None:
    west, south, east, north = (round(float(v), 4) for v in bbox)
    st.session_state["bbox_west"] = west
    st.session_state["bbox_south"] = south
    st.session_state["bbox_east"] = east
    st.session_state["bbox_north"] = north


def _init_session_state() -> None:
    default_region = "cordoba_argentina"
    if "selected_region" not in st.session_state:
        st.session_state["selected_region"] = default_region
    if "last_preset" not in st.session_state:
        st.session_state["last_preset"] = default_region
    if "bbox_west" not in st.session_state:
        _apply_bbox_to_state(REGION_PRESETS[default_region]["bbox"])
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


def _compact_status(status: str) -> str:
    """Shorten climatology status labels so 4-column KPI cards do not clip."""
    mapping = {
        "Critical Anomaly (> +1.5σ)": "Крит. (> +1.5σ)",
        "Extreme Anomaly (> +1.5σ)": "Крит. (> +1.5σ)",
        "Above Normal (> P75)": "Выше нормы (>P75)",
        "Below Normal (< P25)": "Ниже нормы (<P25)",
        "Normal (P25–P75)": "Норма (P25–P75)",
    }
    return mapping.get(status, status)


# ---------------------------------------------------------------------------
# Plotly Chart Builders
# ---------------------------------------------------------------------------
def _make_calendar_heatmap(
    calendar_df: pd.DataFrame,
    colorscale: str,
    metric_label: str,
) -> go.Figure:
    years = [str(y) for y in calendar_df.index]
    z_vals = calendar_df.to_numpy(dtype=float)
    show_text = len(years) <= 30
    cb_title = "Очаги" if "очаг" in metric_label.lower() else "МВт"
    clean_ints = np.nan_to_num(np.round(z_vals, 0), nan=0.0).astype(int).astype(str)
    text_matrix = np.where(np.isnan(z_vals), "", clean_ints) if show_text else None

    heatmap = go.Heatmap(
        z=z_vals,
        x=list(calendar_df.columns),
        y=years,
        colorscale=colorscale,
        colorbar=dict(title=dict(text=cb_title, side="right"), thickness=14, len=0.9),
        hovertemplate="Год: %{y}<br>Месяц: %{x}<br>" + metric_label + ": %{z:,.1f}<extra></extra>",
        text=text_matrix,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(size=10),
    )
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=dict(
            text=f"Тепловой календарь активности горения (Год × Месяц) — {metric_label}",
            font=dict(size=14),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        xaxis_title="Календарный месяц",
        yaxis_title="Год",
        yaxis=dict(autorange="reversed"),
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
    months = climatology_df["month_name"].tolist()

    fig = go.Figure()
    # P25-P75 interquartile band
    fig.add_trace(
        go.Scatter(
            x=months + months[::-1],
            y=climatology_df["p75"].tolist() + climatology_df["p25"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(100, 149, 237, 0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Исторический размах (P25–P75)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=climatology_df["median"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2.5, dash="dash"),
            name="Историческая медиана",
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
                name=f"Выбранный год ({target_year})",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Сезонный профиль ({target_year} г.)",
            font=dict(size=13),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        xaxis_title="Месяц",
        yaxis_title=metric_label,
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
    col = "anomaly_pct" if mode == "% отклонения от медианы" else "z_score"
    y_title = "Отклонение от медианы, %" if mode == "% отклонения от медианы" else "Z-score (σ)"

    colors = []
    for _, row in year_df.iterrows():
        if row["z_score"] > 1.5:
            colors.append("#d62728")
        elif row["value"] > row["p75"]:
            colors.append("#ff7f0e")
        elif row["value"] < row["p25"]:
            colors.append("#6c757d")
        else:
            colors.append("#2ca02c")

    fig = go.Figure(
        data=[
            go.Bar(
                x=year_df["month_name"],
                y=year_df[col],
                marker_color=colors,
                text=[f"{v:+.1f}" for v in year_df[col]],
                textposition="outside",
                customdata=np.stack(
                    [year_df["value"], year_df["median"], year_df["anomaly_status"]],
                    axis=-1,
                ),
                hovertemplate=(
                    "Месяц: %{x}<br>"
                    + y_title
                    + ": %{y:+.2f}<br>"
                    + "Факт: %{customdata[0]:,.1f} | Медиана: %{customdata[1]:,.1f}<br>"
                    + "Статус: %{customdata[2]}<extra></extra>"
                ),
            )
        ]
    )
    fig.add_hline(y=0, line_width=1.2, line_color="#64748b")
    if mode != "% отклонения от медианы":
        fig.add_hline(
            y=1.5,
            line_dash="dot",
            line_color="#d62728",
            annotation_text="Порог критической аномалии (+1.5σ)",
            annotation_font_color="#f8fafc",
        )
    fig.update_layout(
        title=dict(
            text=f"Аномалии по месяцам ({target_year} г.): {mode}",
            font=dict(size=13),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        xaxis_title="Месяц",
        yaxis_title=y_title,
        height=430,
        margin=dict(l=50, r=20, t=68, b=90),
    )
    return fig


def _make_raw_vs_harmonized_chart(
    monthly_df: pd.DataFrame,
    granularity: str,
    use_frp: bool,
) -> go.Figure:
    if granularity == "Годовая сумма (Annual)":
        df = monthly_df.groupby("year", as_index=False).sum(numeric_only=True)
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
    y_title = "Мощность излучения FRP, МВт" if use_frp else "Количество очагов / точек"

    fig = go.Figure()
    if not use_frp:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[raw_col],
                mode="lines+markers",
                name="ДО: Сырая сумма точек (MODIS + VIIRS)",
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
                name="VIIRS после сетки 0.01° (без α-калибровки)",
                line=dict(color="#ff7f0e", width=2.0, dash="dot"),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[raw_col],
                mode="lines",
                name="ДО: Сырая сумма FRP (MODIS + VIIRS)",
                line=dict(color="#d62728", width=2.2, dash="dash"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[modis_col],
            mode="lines",
            name="Опорный ряд MODIS (1 км)",
            line=dict(color="#17becf", width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[harm_col],
            mode="lines+markers",
            name="ПОСЛЕ: Гармонизированный ряд (MODIS + α·VIIRS)",
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
            text="Запуск VIIRS (2012-03)",
            showarrow=False,
            font=dict(size=12, color="#f8fafc"),
            bgcolor="rgba(15, 23, 42, 0.82)",
            bordercolor="#94a3b8",
            borderpad=3,
        )

    fig.update_layout(
        title=dict(
            text="До и После гармонизации: устранение приборного разрыва 2012 года",
            font=dict(size=14),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        xaxis_title="Период",
        yaxis_title=y_title,
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
        fig.update_layout(title="Оперативное окно: нет данных по датам")
        return fig

    g_daily = (
        gridded_df.groupby("acq_date", as_index=False)
        .agg(
            raw_points=("raw_point_count", "sum"),
            gridded_cells=("active_cell_count", "sum"),
            raw_frp=("frp_sum", "sum"),
        )
        .sort_values("acq_date")
    )
    if (
        harmonized_cells_df is not None
        and not harmonized_cells_df.empty
        and "calibrated_cell_count" in harmonized_cells_df.columns
    ):
        h_daily = harmonized_cells_df.groupby("acq_date", as_index=False).agg(
            harmonized_cells=("calibrated_cell_count", "sum"),
            harmonized_frp=("calibrated_frp", "sum"),
        )
        daily = g_daily.merge(h_daily, on="acq_date", how="outer").fillna(0.0).sort_values("acq_date")
    else:
        daily = g_daily.copy()
        daily["harmonized_cells"] = daily["gridded_cells"]
        daily["harmonized_frp"] = daily["raw_frp"]

    if not use_frp:
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["raw_points"],
                name="ДО: Сырые точки",
                marker_color="#d62728",
            )
        )
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["gridded_cells"],
                name="Сетка 0.01° (без α)",
                marker_color="#ff7f0e",
            )
        )
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["harmonized_cells"],
                name="ПОСЛЕ: Гармонизировано (α·VIIRS)",
                marker_color="#2ca02c",
            )
        )
        y_title = "Очагов / точек за сутки"
    else:
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["raw_frp"],
                name="ДО: Сырой FRP (МВт)",
                marker_color="#d62728",
            )
        )
        fig.add_trace(
            go.Bar(
                x=daily["acq_date"],
                y=daily["harmonized_frp"],
                name="ПОСЛЕ: Гармонизированный FRP",
                marker_color="#2ca02c",
            )
        )
        y_title = "Мощность FRP, МВт"

    fig.update_layout(
        title=dict(
            text="Оперативное окно (Live / <2 лет): подневная динамика гармонизации",
            font=dict(size=14),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        xaxis_title="Дата съёмки (UTC)",
        yaxis_title=y_title,
        barmode="group",
        height=380,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=55, r=25, t=65, b=85),
    )
    return fig


def _make_overlap_scatter(
    overlap_df: pd.DataFrame,
    tm: dict[str, float],
) -> go.Figure:
    valid = overlap_df[
        (overlap_df["modis_active_cells"] > 0) & (overlap_df["viirs_active_cells"] > 0)
    ].copy()
    fig = go.Figure()
    if valid.empty:
        fig.update_layout(title="Недостаточно парных наблюдений (2012–2025) для корреляции")
        return fig

    fig.add_trace(
        go.Scatter(
            x=valid["viirs_active_cells"],
            y=valid["modis_active_cells"],
            mode="markers",
            name="Месяцы перекрытия (2012–2025)",
            marker=dict(
                size=8,
                color=valid["year"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Год", thickness=12, len=0.85),
                opacity=0.82,
            ),
            text=[f"{int(y)}-{int(m):02d}" for y, m in zip(valid["year"], valid["month"])],
            hovertemplate="Период: %{text}<br>VIIRS ячеек: %{x:,.0f}<br>MODIS ячеек: %{y:,.0f}<extra></extra>",
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
                f"Перекрытие (2012–2025): "
                f"R²={tm.get('overlap_r2', 0.0):.4f}, r={tm.get('overlap_pearson_r', 0.0):.4f}"
            ),
            font=dict(size=13),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        xaxis_title="VIIRS активные ячейки 0.01° (шт/мес)",
        yaxis_title="MODIS активные ячейки 0.01° (шт/мес)",
        height=430,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=68, b=90),
    )
    return fig


def _make_alpha_bar_chart(calibration_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for dn, label, color in [
        ("D", "День (D)", "#ff7f0e"),
        ("N", "Ночь (N)", "#1f77b4"),
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
                    array=(sub["alpha_ci_high"] - sub["alpha_cells"]).clip(lower=0.0),
                    arrayminus=(sub["alpha_cells"] - sub["alpha_ci_low"]).clip(lower=0.0),
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Коэффициенты α(m, D/N) с 95% ДИ",
            font=dict(size=13),
            x=0.01,
            xanchor="left",
            y=0.96,
        ),
        xaxis_title="Месяц",
        yaxis_title="Коэффициент приведения α",
        barmode="group",
        height=430,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=68, b=90),
    )
    return fig


# ---------------------------------------------------------------------------
# Folium Geospatial Map Builder
# ---------------------------------------------------------------------------
class _SafeLayerControl(folium.LayerControl):
    """LayerControl using `var` and viewport-aware Leaflet resize hooks for tabs."""

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
                var _center = _map.getCenter();
                var _zoom = _map.getZoom();
                var _lastWidth = 0;
                var _refreshMapSize = function() {
                    var el = _map.getContainer();
                    if (!el) return;
                    var w = el.offsetWidth;
                    if (w > 0 && w !== _lastWidth) {
                        var firstShow = (_lastWidth === 0);
                        _lastWidth = w;
                        _map.invalidateSize(false);
                        if (firstShow) {
                            _map.setView(_center, _zoom, {animate: false});
                        }
                    }
                };
                if (typeof ResizeObserver !== "undefined") {
                    new ResizeObserver(_refreshMapSize).observe(_map.getContainer());
                }
                window.addEventListener("resize", _refreshMapSize);
                setInterval(_refreshMapSize, 250);
                _map.on("draw:created", function(e) {
                    if (e && e.layer && e.layer.getBounds) {
                        var b = e.layer.getBounds();
                        var sw = b.getSouthWest();
                        var ne = b.getNorthEast();
                        var msg = "BBox: west=" + sw.lng.toFixed(3) +
                            ", south=" + sw.lat.toFixed(3) +
                            ", east=" + ne.lng.toFixed(3) +
                            ", north=" + ne.lat.toFixed(3);
                        e.layer.bindPopup(msg).openPopup();
                    }
                });
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

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="OpenStreetMap")

    # Active Bounding Box polygon
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#21918c",
        weight=2.5,
        fill=True,
        fill_opacity=0.05,
        tooltip=f"Active BBox: [{west:.2f}, {south:.2f}, {east:.2f}, {north:.2f}]",
    ).add_to(fmap)

    # Drawing tool for custom user BBox selection
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": True,
            "rectangle": True,
        },
        edit_options={"edit": False},
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

        modis_fg = folium.FeatureGroup(name="MODIS 1 km ячейки", show=True)
        viirs_fg = folium.FeatureGroup(name="VIIRS 375 м → 1 km ячейки", show=True)

        for row in sample_df.itertuples(index=False):
            is_modis = row.sensor_group == "MODIS"
            target_group = modis_fg if is_modis else viirs_fg
            color = "#0077b6" if is_modis else "#e85d04"
            radius = float(np.clip(2.5 + np.log1p(row.frp_sum) * 0.9, 3.0, 9.0))
            popup_html = (
                f"<b>Сенсор:</b> {row.sensor_group} ({row.daynight})<br>"
                f"<b>Дата:</b> {row.acq_date}<br>"
                f"<b>Ячейка:</b> {row.cell_center_lat:.3f}, {row.cell_center_lon:.3f}<br>"
                f"<b>Сырых точек в ячейке:</b> {int(row.raw_point_count)}<br>"
                f"<b>FRP (сумма):</b> {row.frp_sum:.1f} МВт"
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
                name="Тепловая плотность FRP",
                radius=14,
                blur=16,
                min_opacity=0.35,
                show=True,
            ).add_to(fmap)

    _SafeLayerControl(collapsed=False).add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session_state()
    st.markdown(
        """
        <style>
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
        "Устранение приборного разрыва 2012 г. между MODIS (1 км) и VIIRS (375 м) "
        "с формированием согласованного ряда 2001–2026 гг."
    )

    # -----------------------------------------------------------------------
    # Sidebar Controls
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Панель управления")

        source_mode = st.radio(
            "Источник данных",
            options=[
                "Демо-архив (2001–2026, офлайн)",
                "Загрузка CSV (FIRMS Archive)",
                "Живой запрос NASA FIRMS Area API",
            ],
            index=0,
        )

        st.subheader("🌍 Регион и Bounding Box")
        preset_keys = list(REGION_PRESETS.keys())
        region_key = st.selectbox(
            "Пресет региона",
            options=preset_keys,
            format_func=lambda k: REGION_PRESETS[k]["name_ru"],
            key="selected_region",
        )
        if region_key != st.session_state["last_preset"]:
            st.session_state["last_preset"] = region_key
            _apply_bbox_to_state(REGION_PRESETS[region_key]["bbox"])

        col_w, col_s = st.columns(2)
        west = col_w.number_input("Min Lon (West)", step=0.1, format="%.3f", key="bbox_west")
        south = col_s.number_input("Min Lat (South)", step=0.1, format="%.3f", key="bbox_south")
        col_e, col_n = st.columns(2)
        east = col_e.number_input("Max Lon (East)", step=0.1, format="%.3f", key="bbox_east")
        north = col_n.number_input("Max Lat (North)", step=0.1, format="%.3f", key="bbox_north")

        if st.button("Сбросить BBox к пресету", width="stretch"):
            st.session_state["pending_bbox"] = REGION_PRESETS[region_key]["bbox"]
            st.rerun()

        active_bbox = (min(west, east), min(south, north), max(west, east), max(south, north))

        # Source-specific input widgets
        uploaded_files = None
        if source_mode == "Загрузка CSV (FIRMS Archive)":
            st.info(
                f"Загрузите CSV из [FIRMS Archive Download]({FIRMS_ARCHIVE_URL}) "
                "(MODIS и/или VIIRS)."
            )
            uploaded_files = st.file_uploader(
                "Выберите CSV-файлы FIRMS",
                type=["csv"],
                accept_multiple_files=True,
            )

        elif source_mode == "Живой запрос NASA FIRMS Area API":
            st.info(f"Получите бесплатный MAP_KEY на [FIRMS Map Key]({FIRMS_MAP_KEY_URL}).")
            default_key = ""
            try:
                default_key = str(st.secrets.get("FIRMS_MAP_KEY", "") or "").strip()
            except Exception:
                default_key = ""
            if not default_key:
                try:
                    client_env = FIRMSApiClient()
                    default_key = client_env.map_key
                except Exception:
                    default_key = ""
            api_key_input = st.text_input("FIRMS MAP_KEY", value=default_key, type="password")
            api_client = FIRMSApiClient(map_key=api_key_input)

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

            if st.button("Проверить покрытие API (Data Availability)", width="stretch"):
                try:
                    avail_df = api_client.check_data_availability("ALL")
                    st.session_state["api_availability_df"] = avail_df
                    st.success(f"Подключение подтверждено ({len(avail_df)} сенсоров).")
                except FIRMSApiError as exc:
                    st.error(str(exc))

            if not st.session_state["api_availability_df"].empty:
                with st.expander("Доступность сенсоров FIRMS"):
                    st.dataframe(st.session_state["api_availability_df"], hide_index=True)

            api_modis_src = st.selectbox("Источник MODIS", ["MODIS_NRT", "MODIS_SP"], index=0)
            api_viirs_src = st.selectbox(
                "Источник VIIRS",
                ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_SP", "VIIRS_SNPP_SP"],
                index=0,
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
                st.caption(f"Доступный диапазон FIRMS: {bounds_text}")

            api_days = st.slider("Период (дней, чанки по 10)", min_value=1, max_value=30, value=5)
            api_end_default = common_max_dt or date.today()
            api_start_default = api_end_default - timedelta(days=int(api_days) - 1)
            if common_min_dt and api_start_default < common_min_dt:
                api_start_default = common_min_dt

            api_start_date = st.date_input(
                "Дата начала выборки",
                value=api_start_default,
                min_value=common_min_dt,
                max_value=common_max_dt or date.today(),
                key=f"api_start_{api_modis_src}_{api_viirs_src}",
            )

            if st.button("Загрузить данные через FIRMS API", type="primary", width="stretch"):
                try:
                    req_end = api_start_date + timedelta(days=int(api_days) - 1)
                    m_max = per_src_bounds.get(api_modis_src, (None, req_end))[1] or req_end
                    v_max = per_src_bounds.get(api_viirs_src, (None, req_end))[1] or req_end
                    m_end = min(req_end, m_max)
                    v_end = min(req_end, v_max)
                    m_start = min(api_start_date, m_end)
                    v_start = min(api_start_date, v_end)

                    with st.spinner("Загрузка данных из NASA LANCE FIRMS..."):
                        m_api = api_client.fetch_area_range(
                            source=api_modis_src,
                            bbox=active_bbox,
                            start_date=m_start.strftime("%Y-%m-%d"),
                            end_date=m_end.strftime("%Y-%m-%d"),
                        )
                        v_api = api_client.fetch_area_range(
                            source=api_viirs_src,
                            bbox=active_bbox,
                            start_date=v_start.strftime("%Y-%m-%d"),
                            end_date=v_end.strftime("%Y-%m-%d"),
                        )
                    st.session_state["api_modis_df"] = m_api
                    st.session_state["api_viirs_df"] = v_api
                    st.success(
                        f"Загружено: MODIS={len(m_api)}, VIIRS={len(v_api)} точек "
                        f"({m_start.isoformat()}..{max(m_end, v_end).isoformat()})."
                    )
                except FIRMSApiError as exc:
                    st.error(str(exc))

        st.subheader("🔬 Фильтры и калибровка")
        year_min, year_max = st.slider(
            "Диапазон лет",
            min_value=YEAR_RANGE[0],
            max_value=YEAR_RANGE[1],
            value=YEAR_RANGE,
        )
        daynight_option = st.selectbox(
            "Время суток пролёта (daynight)",
            options=["All", "D", "N"],
            format_func=lambda x: {"All": "Все (День + Ночь)", "D": "Только день (D)", "N": "Только ночь (N)"}[x],
        )
        modis_min_conf = st.slider(
            "Мин. достоверность MODIS (0–100)",
            min_value=0,
            max_value=100,
            value=DEFAULT_MODIS_MIN_CONFIDENCE,
            step=5,
        )
        viirs_conf = st.multiselect(
            "Допустимая достоверность VIIRS",
            options=["l", "n", "h"],
            default=DEFAULT_VIIRS_CONFIDENCE,
        )
        combo_mode = st.selectbox(
            "Режим сшивки после 2012 г.",
            options=["calibrated_viirs", "ensemble"],
            format_func=lambda m: (
                "Откалиброванный VIIRS × α (бесшовное продление)"
                if m == "calibrated_viirs"
                else "Ансамбль (MODIS + VIIRS × α) / 2"
            ),
        )
        metric_choice = st.radio(
            "Показатель календаря",
            options=["harmonized_hotspots", "harmonized_frp"],
            format_func=lambda m: (
                "Гармонизированные очаги (ячейки 1 км)"
                if m == "harmonized_hotspots"
                else "Мощность излучения FRP (МВт)"
            ),
        )
        palette_name = st.selectbox(
            "Цветовая палитра (CVD-friendly)",
            options=list(COLOR_PALETTES.keys()),
            format_func=lambda p: COLOR_PALETTES[p]["label"],
        )

    # -----------------------------------------------------------------------
    # Load Selected Data Source & Execute Cached Harmonization + Climatology
    # -----------------------------------------------------------------------
    if not viirs_conf:
        st.warning("Выберите хотя бы одну категорию достоверности VIIRS (`n`, `h` или `l`).")
        st.stop()

    start_date_str = f"{year_min:04d}-01-01"
    end_date_str = f"{year_max:04d}-12-31"
    viirs_conf_tuple = tuple(sorted(viirs_conf))

    if source_mode == "Демо-архив (2001–2026, офлайн)":
        active_region_label = REGION_PRESETS[region_key]["name_ru"]
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
        )
        if result.monthly_df.empty:
            st.warning(
                "В выбранном регионе и диапазоне дат не найдено очагов, прошедших фильтры достоверности. "
                "Попробуйте расширить Bounding Box, диапазон лет или снизить порог confidence."
            )
            st.stop()
    else:
        if source_mode == "Загрузка CSV (FIRMS Archive)":
            if not uploaded_files:
                st.warning("Загрузите один или несколько CSV-файлов MODIS/VIIRS в боковой панели, чтобы начать расчёт.")
                st.stop()
            raw_modis_df, raw_viirs_df = load_csv_dataset(uploaded_files)
            active_region_label = "Пользовательский CSV-архив FIRMS"
        else:
            raw_modis_df = st.session_state["api_modis_df"]
            raw_viirs_df = st.session_state["api_viirs_df"]
            if raw_modis_df.empty and raw_viirs_df.empty:
                st.info(
                    "Нажмите **«Загрузить данные через FIRMS API»** в боковой панели, чтобы получить "
                    "актуальные данные NASA FIRMS по выбранному региону, или переключитесь на демо-архив."
                )
                st.stop()
            active_region_label = f"Живой API FIRMS ({REGION_PRESETS[region_key]['name_ru']})"

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
        needs_reference_cal = (
            source_mode == "Живой запрос NASA FIRMS Area API"
            or unique_years_count < 2
        )

        ref_result = None
        ext_cal_df = None
        ext_overlap_df = None
        ref_tm = None
        if needs_reference_cal:
            ref_result, ext_cal_df = _load_regional_reference_cached(
                region_key=region_key,
                modis_min_conf=modis_min_conf,
                viirs_allowed_conf=viirs_conf_tuple,
                combination_mode=combo_mode,
                daynight_filter=daynight_option,
            )
            ext_overlap_df = ref_result.overlap_monthly_df
            ref_tm = ref_result.transition_metrics

        result = _run_harmonization_cached(
            modis_df=modis_filtered,
            viirs_df=viirs_filtered,
            modis_min_conf=modis_min_conf,
            viirs_allowed_conf=viirs_conf_tuple,
            combination_mode=combo_mode,
            daynight_filter=daynight_option,
            external_calibration_df=ext_cal_df,
            external_overlap_df=ext_overlap_df,
            reference_transition_metrics=ref_tm,
        )

        if result.monthly_df.empty:
            st.warning(
                "В выбранном регионе и диапазоне дат не найдено очагов, прошедших фильтры достоверности. "
                "Попробуйте расширить Bounding Box, диапазон лет или снизить порог confidence."
            )
            st.stop()

        calendar_df, climatology_df, anomalies_df, kpis = _compute_climatology_bundle_cached(
            monthly_df=result.monthly_df,
            gridded_df=result.gridded_df,
            transition_metrics=result.transition_metrics,
            ref_monthly_df=ref_result.monthly_df if ref_result is not None else None,
            metric_choice=metric_choice,
        )

    use_frp = metric_choice == "harmonized_frp"
    metric_label = "Гармонизированный FRP (МВт)" if use_frp else "Гармонизированные очаги (ячейки 1 км)"
    tm = result.transition_metrics
    is_ref_mode = bool(tm.get("used_reference_calibration", False))

    area_km2 = _compute_bbox_area_km2(active_bbox)
    st.markdown(
        f"**Регион:** {active_region_label} · **Площадь BBox:** {area_km2:,.0f} км² · "
        f"**Точек после фильтрации:** MODIS `{modis_count:,}` | VIIRS `{viirs_count:,}`"
    )
    if is_ref_mode:
        st.info(
            f"ℹ️ **Режим оперативного мониторинга (< 2 лет / NASA FIRMS NRT API):** "
            f"применены предрассчитанные региональные коэффициенты `α(m, daynight)` из "
            f"`data/processed/calibration_alpha_{region_key}.csv` и опорная климатология 2001–2025 гг. "
            f"(в текущем окне: `{tm.get('live_raw_points', 0):.0f}` сырых точек → "
            f"`{tm.get('live_gridded_cells', 0):.0f}` ячеек 0.01° → "
            f"`{tm.get('live_harmonized_hotspots', 0):.1f}` гармонизированных очагов)."
        )

    # -----------------------------------------------------------------------
    # Top KPI Row
    # -----------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Пиковый месяц",
        kpis.get("peak_month_name", "—"),
        delta=f"Медиана: {kpis.get('peak_month_median', 0):,.1f}",
        delta_color="off",
    )
    if is_ref_mode:
        k2.metric(
            "Очагов в окне (Live)",
            f"{tm.get('live_harmonized_hotspots', 0):.1f}",
            delta=f"Сырых: {tm.get('live_raw_points', 0):.0f} · Ячеек: {tm.get('live_gridded_cells', 0):.0f}",
            delta_color="off",
        )
    else:
        worst_frp = float(kpis.get("worst_year_frp", 0.0))
        frp_text = f"{worst_frp / 1000.0:.1f} ГВт" if worst_frp >= 1000.0 else f"{worst_frp:,.0f} МВт"
        k2.metric(
            "Худший год",
            str(kpis.get("worst_year", "—")),
            delta=f"{kpis.get('worst_year_hotspots', 0):,.0f} яч. · {frp_text}",
            delta_color="off",
        )
    k3.metric(
        f"Сезон {kpis.get('focus_year', '')} ({kpis.get('focus_month_name', '')})",
        f"{kpis.get('focus_anomaly_pct', 0):+.1f}%",
        delta=f"{kpis.get('focus_z_score', 0):+.2f}σ · {_compact_status(kpis.get('focus_status', ''))}",
        delta_color="inverse",
    )
    k4.metric(
        "Коэфф. α (VIIRS)",
        f"{tm.get('mean_alpha_overall', 1.0):.3f}",
        delta=f"-{tm.get('jump_reduction_pct', 0):.0f}% скачок · D:{tm.get('mean_alpha_day', 1.0):.2f} N:{tm.get('mean_alpha_night', 1.0):.2f}",
        delta_color="off",
        help="Средний коэффициент приведения VIIRS (375 м) → MODIS (1 км)",
    )

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        [
            "📅 Календарь активности горения",
            "📊 Доказательство гармонизации (Raw vs Harmonized)",
            "🗺️ Карта очагов и экспорт CSV",
        ]
    )

    # === TAB 1: BURNING ACTIVITY CALENDAR ===
    with tab1:
        st.plotly_chart(
            _make_calendar_heatmap(
                calendar_df=calendar_df,
                colorscale=COLOR_PALETTES[palette_name]["plotly"],
                metric_label=metric_label,
            ),
            width="stretch",
        )

        available_years = sorted(anomalies_df["year"].unique().tolist(), reverse=True)
        default_cmp_year = kpis.get("worst_year", available_years[0])
        if default_cmp_year not in available_years:
            default_cmp_year = available_years[0]

        c_left, c_right = st.columns([1, 2])
        with c_left:
            selected_year = st.selectbox(
                "Год для сравнения с исторической нормой",
                options=available_years,
                index=available_years.index(default_cmp_year),
            )
        with c_right:
            anom_metric_mode = st.radio(
                "Метрика аномалии",
                options=["% отклонения от медианы", "Z-score (σ)"],
                horizontal=True,
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
            )
        with p_col2:
            st.plotly_chart(
                _make_anomaly_bar_chart(
                    anomalies_df=anomalies_df,
                    target_year=selected_year,
                    mode=anom_metric_mode,
                ),
                width="stretch",
            )

        with st.expander(f"Детальная таблица аномалий за {selected_year} год"):
            year_table = anomalies_df[anomalies_df["year"] == selected_year][
                [
                    "date_label",
                    "month_name",
                    "value",
                    "median",
                    "p25",
                    "p75",
                    "anomaly_abs",
                    "anomaly_pct",
                    "z_score",
                    "anomaly_status",
                    "primary_source",
                ]
            ].rename(
                columns={
                    "date_label": "Период",
                    "month_name": "Месяц",
                    "value": "Гармонизировано",
                    "median": "Медиана (2001–2025)",
                    "p25": "P25",
                    "p75": "P75",
                    "anomaly_abs": "Отклонение (абс.)",
                    "anomaly_pct": "Отклонение (%)",
                    "z_score": "Z-score",
                    "anomaly_status": "Статус",
                    "primary_source": "Источник ряда",
                }
            )
            st.dataframe(year_table, hide_index=True, width="stretch")

    # === TAB 2: HARMONIZATION PROOF ===
    with tab2:
        if is_ref_mode and ref_result is not None:
            st.plotly_chart(
                _make_live_daily_chart(
                    gridded_df=result.gridded_df,
                    use_frp=use_frp,
                    harmonized_cells_df=result.harmonized_cells_df,
                ),
                width="stretch",
            )
            proof_monthly_df = ref_result.monthly_df
        else:
            proof_monthly_df = result.monthly_df

        granularity = st.radio(
            "Детализация временного ряда",
            options=["Годовая сумма (Annual)", "Помесячный ряд (Monthly)"],
            horizontal=True,
        )
        st.plotly_chart(
            _make_raw_vs_harmonized_chart(
                monthly_df=proof_monthly_df,
                granularity=granularity,
                use_frp=use_frp,
            ),
            width="stretch",
        )

        if is_ref_mode:
            st.caption(
                f"Метрики разрыва 2012 г. и корреляция перекрытия ниже приведены по опорному "
                f"архиву региона `{REGION_PRESETS[region_key]['name_ru']}` (2001–2026), так как "
                f"активная выборка охватывает менее 2 лет и калибруется по `calibration_alpha_{region_key}.csv`."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Сырой скачок после 2012 г.",
            f"{tm.get('raw_combined_jump_pct', 0):+.1f}%",
            delta=f"{tm.get('pre2012_raw_annual_mean', 0):,.0f} → {tm.get('post2012_raw_combined_annual_mean', 0):,.0f} точек/год",
            delta_color="inverse",
        )
        m2.metric(
            "Сетка 0.01° без α",
            f"{tm.get('gridded_uncalibrated_jump_pct', 0):+.1f}%",
            delta=f"{tm.get('pre2012_gridded_annual_mean', 0):,.0f} → {tm.get('post2012_gridded_viirs_annual_mean', 0):,.0f} ячеек/год",
            delta_color="inverse",
        )
        m3.metric(
            "Остаточный сдвиг сенсоров (2012–2025)",
            f"{tm.get('overlap_sensor_residual_bias_pct', 0):+.2f}%",
            delta="VIIRS·α vs MODIS на перекрытии",
        )
        m4.metric(
            "Подавление скачка 2012",
            f"{tm.get('jump_reduction_pct', 0):.2f}%",
            delta=f"Overlap R² = {tm.get('overlap_r2', 0):.4f}",
        )

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.plotly_chart(
                _make_overlap_scatter(result.overlap_monthly_df, tm),
                width="stretch",
            )
        with s_col2:
            st.plotly_chart(
                _make_alpha_bar_chart(result.calibration_df),
                width="stretch",
            )

        with st.expander("Таблица калибровочных коэффициентов α(m, daynight) и 95% ДИ (бутстрап)"):
            st.dataframe(result.calibration_df, hide_index=True, width="stretch")

    # === TAB 3: GEOSPATIAL EXPLORER & CSV EXPORT ===
    with tab3:
        st.markdown(
            f"На карте можно рисовать прямоугольник или полигон инструментом **Draw** слева "
            f"(клик по фигуре покажет координаты BBox), включать внешние слои OGC WMS "
            f"([CONAE]({CONAE_CATALOG_OGC_URL}), [CONAE Focos]({CONAE_FOCOS_URL}), "
            f"[NASA FIRMS]({FIRMS_PORTAL_URL}) / [Global Fire Map]({FIRMS_MAP_URL}))."
        )

        f1, f2, f3 = st.columns(3)
        map_years = ["Все годы"] + sorted(result.gridded_df["year"].unique().tolist(), reverse=True)
        default_map_year = kpis.get("worst_year")
        default_map_idx = (
            map_years.index(default_map_year)
            if default_map_year in map_years
            else (1 if len(map_years) > 1 else 0)
        )
        map_year_sel = f1.selectbox(
            "Год на карте",
            options=map_years,
            index=default_map_idx,
            key=f"map_year_{region_key}_{source_mode}",
        )
        map_month_sel = f2.selectbox(
            "Месяц на карте",
            options=["Все месяцы"] + list(range(1, 13)),
            format_func=lambda m: m if isinstance(m, str) else f"{m:02d} — {MONTH_LABELS[m - 1]}",
        )
        max_markers = f3.slider("Макс. ячеек на карте", min_value=100, max_value=1500, value=300, step=100)

        map_df = result.gridded_df.copy()
        if map_year_sel != "Все годы":
            map_df = map_df[map_df["year"] == int(map_year_sel)]
        if map_month_sel != "Все месяцы":
            map_df = map_df[map_df["month"] == int(map_month_sel)]

        fmap = _build_folium_map(map_df, active_bbox, max_points=max_markers)
        map_html = fmap.get_root().render()
        if hasattr(st, "iframe"):
            st.iframe(map_html, height=540)
        else:
            import streamlit.components.v1 as components

            components.html(map_html, height=540, scrolling=False)

        st.divider()
        st.subheader("📥 Экспорт гармонизированных данных (CSV)")
        d1, d2 = st.columns(2)
        d3, d4 = st.columns(2)

        d1.download_button(
            label="harmonized_hotspots.csv",
            data=result.harmonized_cells_df.to_csv(index=False).encode("utf-8"),
            file_name=f"harmonized_hotspots_{region_key}.csv",
            mime="text/csv",
            width="stretch",
        )
        d2.download_button(
            label="burning_activity_calendar.csv",
            data=calendar_df.to_csv().encode("utf-8"),
            file_name=f"burning_activity_calendar_{region_key}.csv",
            mime="text/csv",
            width="stretch",
        )
        d3.download_button(
            label="harmonized_monthly_series.csv",
            data=anomalies_df.to_csv(index=False).encode("utf-8"),
            file_name=f"harmonized_monthly_series_{region_key}.csv",
            mime="text/csv",
            width="stretch",
        )
        d4.download_button(
            label="calibration_alpha_table.csv",
            data=result.calibration_df.to_csv(index=False).encode("utf-8"),
            file_name=f"calibration_alpha_{region_key}.csv",
            mime="text/csv",
            width="stretch",
        )


if __name__ == "__main__":
    main()
