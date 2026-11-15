"""Comprehensive QA and audit compliance tests for data loading, calibration, and analytics."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests

from src.analytics import (
    build_calendar_matrix,
    check_series_nonstationarity,
    compute_critical_fire_windows,
    compute_monthly_anomalies,
    compute_monthly_climatology,
)
from src.config import validate_bbox
from src.data_loader import (
    FIRMSApiClient,
    FIRMSApiError,
    load_csv_dataset_with_report,
)
from src.harmonizer import HarmonizationPipeline, build_daily_sensor_breakdown


class _FakeUpload(BytesIO):
    """Minimal Streamlit UploadedFile stand-in with .name and .size attributes."""

    def __init__(self, raw_bytes: bytes, name: str) -> None:
        super().__init__(raw_bytes)
        self.name = name
        self.size = len(raw_bytes)


class _DummyResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} Error")
            err.response = self  # type: ignore[assignment]
            raise err


# ---------------------------------------------------------------------------
# 1. FIRMS API Validation, Cache Hygiene, Backoff & 5-day Chunking
# ---------------------------------------------------------------------------
def test_firms_api_rejects_html_and_invalid_key_without_caching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _mock_html(*args: Any, **kwargs: Any) -> _DummyResponse:
        return _DummyResponse(200, "<html><body>Service Maintenance</body></html>")

    monkeypatch.setattr(requests, "get", _mock_html)
    client = FIRMSApiClient(map_key="TEST_KEY", cache_dir=tmp_path, max_retries=1)
    with pytest.raises(FIRMSApiError):
        client.fetch_area_csv("VIIRS_SNPP_NRT", (-65.0, -32.0, -64.0, -31.0), 5, "2025-08-01")
    assert list(tmp_path.glob("firms_*.csv")) == []

    def _mock_bad_key(*args: Any, **kwargs: Any) -> _DummyResponse:
        return _DummyResponse(200, "Invalid MAP_KEY.")

    monkeypatch.setattr(requests, "get", _mock_bad_key)
    with pytest.raises(FIRMSApiError):
        client.fetch_area_csv("VIIRS_SNPP_NRT", (-65.0, -32.0, -64.0, -31.0), 5, "2025-08-01")
    assert list(tmp_path.glob("firms_*.csv")) == []


def test_firms_api_rate_limit_retries_with_exponential_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []
    valid_csv = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        "-31.4,-64.2,330.0,0.4,0.4,2025-08-01,1730,N,VIIRS,n,2.0NRT,295.0,12.5,D\n"
    )

    def _mock_get(*args: Any, **kwargs: Any) -> _DummyResponse:
        calls["n"] += 1
        if calls["n"] < 3:
            return _DummyResponse(200, "Exceeding allowed transaction limit.")
        return _DummyResponse(200, valid_csv)

    monkeypatch.setattr(requests, "get", _mock_get)
    client = FIRMSApiClient(
        map_key="TEST_KEY",
        cache_dir=tmp_path,
        max_retries=4,
        backoff_base=0.25,
        sleep_fn=sleeps.append,
    )
    df = client.fetch_area_csv("VIIRS_SNPP_NRT", (-65.0, -32.0, -64.0, -31.0), 5, "2025-08-01")
    assert len(df) == 1
    assert calls["n"] == 3
    assert sleeps == [0.25, 0.5]
    assert len(list(tmp_path.glob("firms_*.csv"))) == 1


def test_firms_api_header_only_zero_rows_allowed_and_poisoned_cache_evicted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Poisoned cache file must be detected, deleted, and replaced on fresh fetch
    cache_file = tmp_path / "firms_MODIS_NRT_-65.0_-32.0_-64.0_-31.0_d5_2025-01-01.csv"
    cache_file.write_text("Exceeding allowed transaction limit.", encoding="utf-8")

    header_only_csv = (
        "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,confidence,version,bright_t31,frp,daynight\n"
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: _DummyResponse(200, header_only_csv))

    client = FIRMSApiClient(map_key="TEST_KEY", cache_dir=tmp_path, max_retries=1)
    df = client.fetch_area_csv("MODIS_NRT", (-65.0, -32.0, -64.0, -31.0), 5, "2025-01-01", use_cache=True)
    assert df.empty
    assert "acq_date" in df.columns
    assert "latitude" in cache_file.read_text(encoding="utf-8")


def test_firms_api_range_chunks_5_days_and_surfaces_partial_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_spans: list[int] = []
    good_csv = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        "-31.4,-64.2,330.0,0.4,0.4,2025-08-01,1730,N,VIIRS,n,2.0NRT,295.0,12.5,D\n"
    )

    def _mock_get(url: str, *args: Any, **kwargs: Any) -> _DummyResponse:
        parts = url.rstrip("/").split("/")
        span = int(parts[-2])
        requested_spans.append(span)
        if parts[-1] == "2025-08-06":
            return _DummyResponse(500, "Internal Server Error")
        return _DummyResponse(200, good_csv)

    monkeypatch.setattr(requests, "get", _mock_get)
    client = FIRMSApiClient(map_key="TEST_KEY", cache_dir=tmp_path, max_retries=1)
    df = client.fetch_area_range(
        source="VIIRS_SNPP_NRT",
        bbox=(-65.0, -32.0, -64.0, -31.0),
        start_date="2025-08-01",
        end_date="2025-08-12",
        use_cache=False,
        allow_partial=True,
    )
    assert requested_spans == [5, 5, 2]
    assert len(df) == 1
    assert len(client.last_failed_windows) == 1
    assert client.last_failed_windows[0]["start_date"] == "2025-08-06"


# ---------------------------------------------------------------------------
# 2. CSV Uploader Diagnostics, Delimiter & Date Parsing
# ---------------------------------------------------------------------------
def test_csv_loader_diagnostics_semicolon_dayfirst_and_error_reporting() -> None:
    semicolon_csv = (
        "latitude;longitude;brightness;scan;track;acq_date;acq_time;satellite;confidence;version;bright_t31;frp;daynight\n"
        "-31,4;-64,2;320.0;1.0;1.0;15/08/2020;1700;Aqua;85;6.1;290.0;45.2;D\n"
    ).replace("-31,4;-64,2", "-31.4;-64.2")
    upload = _FakeUpload(semicolon_csv.encode("utf-8"), "modis_eu.csv")

    m_df, v_df, report = load_csv_dataset_with_report(upload, dayfirst=True)
    assert v_df.empty
    assert len(m_df) == 1
    assert m_df.iloc[0]["acq_date"] == "2020-08-15"
    assert report["rows_read"] == 1
    assert report["rows_kept"] == 1
    assert ";" in report["delimiters"]

    bad_upload = _FakeUpload(b"lat,lon,date\n1,2,2020-01-01\n", "bad.csv")
    with pytest.raises(ValueError) as exc_info:
        load_csv_dataset_with_report(bad_upload)
    assert "latitude" in str(exc_info.value) and "Found columns" in str(exc_info.value)

    empty_upload = _FakeUpload(b"", "empty.csv")
    with pytest.raises(ValueError):
        load_csv_dataset_with_report(empty_upload)


# ---------------------------------------------------------------------------
# 3. Sum-Ratio Calibration Conservation, Zero-Overlap & No-Data Masking
# ---------------------------------------------------------------------------
def test_harmonizer_sum_ratio_exact_conservation_and_no_data_gaps() -> None:
    # Build paired MODIS (2010, 2013-2015) and VIIRS (2013-2015) with year 2011 completely absent
    m_rows = []
    v_rows = []
    for year in [2010, 2013, 2014, 2015]:
        for month in range(1, 13):
            dt = f"{year:04d}-{month:02d}-15"
            for i in range(4):
                m_rows.append(
                    {
                        "latitude": -31.5 + i * 0.05,
                        "longitude": -64.5,
                        "acq_date": dt,
                        "acq_time": "1700",
                        "satellite": "Aqua",
                        "confidence": 80,
                        "frp": 30.0,
                        "daynight": "D",
                    }
                )
            if year >= 2013:
                for i in range(8):
                    v_rows.append(
                        {
                            "latitude": -31.5 + (i % 4) * 0.05 + (0.02 if i >= 4 else 0.0),
                            "longitude": -64.5 + (0.02 if i >= 4 else 0.0),
                            "acq_date": dt,
                            "acq_time": "1715",
                            "satellite": "N",
                            "confidence": "h",
                            "frp": 12.0,
                            "daynight": "D",
                        }
                    )

    m_df = pd.DataFrame(m_rows)
    v_df = pd.DataFrame(v_rows)
    res = HarmonizationPipeline().run(m_df, v_df)
    tm = res.transition_metrics

    assert tm["has_valid_overlap"] is True
    # Exact sum-ratio normalization guarantees near-zero in-sample overlap residual bias
    assert abs(tm["overlap_sensor_residual_bias_pct"]) <= 0.5

    # Unobserved year 2011 must carry data_status == "no_data" and stay NaN in the calendar matrix
    yr2011 = res.monthly_df[res.monthly_df["year"] == 2011]
    assert not yr2011.empty
    assert (yr2011["data_status"] == "no_data").all()
    assert yr2011["harmonized_hotspots"].isna().all()

    cal_mat = build_calendar_matrix(res.monthly_df, value_col="harmonized_hotspots")
    assert cal_mat.loc[2011].isna().all()

    # Climatology must exclude no_data months and report exact count of observed years (4, not 5)
    clim_df = compute_monthly_climatology(res.monthly_df, value_col="harmonized_hotspots")
    assert (clim_df["n_years"] == 4).all()


def test_zero_overlap_emits_nan_ci_and_no_data_status() -> None:
    m_df = pd.DataFrame(
        [
            {
                "latitude": -31.5,
                "longitude": -64.5,
                "acq_date": "2010-08-15",
                "acq_time": "1700",
                "satellite": "Aqua",
                "confidence": 85,
                "frp": 35.0,
                "daynight": "D",
            }
        ]
    )
    v_df = pd.DataFrame(columns=["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "daynight"])
    res = HarmonizationPipeline().run(m_df, v_df)
    assert res.transition_metrics["has_valid_overlap"] is False
    assert (res.calibration_df["calibration_status"] == "no_data").all()
    assert res.calibration_df["alpha_ci_low"].isna().all()
    assert res.calibration_df["alpha_ci_high"].isna().all()


# ---------------------------------------------------------------------------
# 4. Analytics: LOO Robust Z, Non-Stationarity, Critical Fire Windows & BBox
# ---------------------------------------------------------------------------
def test_loo_anomalies_nonstationarity_and_critical_windows() -> None:
    # 12 observed years (2010-2021), full 12 months each with strong upward trend + outlier in 2020-08 + gap in 2015-08
    rows = []
    for idx, y in enumerate(range(2010, 2022)):
        for m in range(1, 13):
            if y == 2015 and m == 8:
                rows.append({"year": y, "month": m, "date_label": f"{y:04d}-{m:02d}", "harmonized_hotspots": np.nan, "harmonized_frp": np.nan, "raw_total_points": 0, "data_status": "no_data"})
                continue
            val = (10.0 + idx * 5.0) if m != 8 else (400.0 if y == 2020 else 50.0 + idx * 10.0)
            rows.append({"year": y, "month": m, "date_label": f"{y:04d}-{m:02d}", "harmonized_hotspots": val, "harmonized_frp": val * 10.0, "raw_total_points": int(val), "data_status": "observed"})

    monthly_df = pd.DataFrame(rows)
    clim = compute_monthly_climatology(monthly_df)
    anom = compute_monthly_anomalies(monthly_df, clim)

    row_2020_aug = anom[(anom["year"] == 2020) & (anom["month"] == 8)].iloc[0]
    assert row_2020_aug["anomaly_status"] == "Critical Anomaly (Robust Z > +1.5)"
    assert row_2020_aug["n_baseline_years"] == 10

    row_2015_aug = anom[(anom["year"] == 2015) & (anom["month"] == 8)].iloc[0]
    assert row_2015_aug["anomaly_status"] == "No Data"

    # Short sample (<8 yrs) -> Insufficient Data
    short_df = monthly_df[monthly_df["year"] <= 2014].copy()
    short_anom = compute_monthly_anomalies(short_df, compute_monthly_climatology(short_df))
    obs_short = short_anom[short_anom["data_status"] == "observed"]
    assert (obs_short["anomaly_status"] == "Insufficient Data (<8 yrs)").all()

    nonstat = check_series_nonstationarity(monthly_df)
    assert nonstat["has_significant_trend"] is True

    # Critical fire windows on daily breakdown (quiet winter/spring + escalating late-August/September peak)
    daily_df = pd.DataFrame(
        [{"acq_date": f"{y:04d}-01-15", "harmonized_cells": 0.5} for y in range(2015, 2021)]
        + [{"acq_date": f"{y:04d}-04-15", "harmonized_cells": 2.0} for y in range(2015, 2021)]
        + [{"acq_date": f"{y:04d}-07-15", "harmonized_cells": 5.0} for y in range(2015, 2021)]
        + [{"acq_date": f"{y:04d}-09-01", "harmonized_cells": 30.0 + (y - 2015) * 5.0} for y in range(2015, 2021)]
    )
    crit = compute_critical_fire_windows(daily_df)
    assert len(crit) == 52
    top_week = crit.sort_values(["p_exceedance", "mean_weekly"], ascending=False).iloc[0]
    assert top_week["week"] in (34, 35, 36)
    assert top_week["p_exceedance"] > 0.5


def test_validate_bbox_checks() -> None:
    ok, _ = validate_bbox((-65.6, -32.6, -63.8, -30.2), is_live_api=True, api_days=5)
    assert ok is True
    bad, msg = validate_bbox((-70.0, -40.0, -50.0, -20.0), is_live_api=True, max_span_deg=10.0)
    assert bad is False and "10" in msg


def test_app_helpers_and_charts_smoke() -> None:
    import app as st_app

    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-65.5, -32.5], [-64.0, -32.5], [-64.0, -30.5], [-65.5, -30.5], [-65.5, -32.5]]],
                },
            }
        ],
    }
    assert st_app._extract_bbox_from_geojson(gj) == (-65.5, -32.5, -64.0, -30.5)
    assert st_app._compact_status("Critical Anomaly (Robust Z > +1.5)") == "Critical (Robust Z>+1.5)"
    assert st_app._compact_status("No Data") == "No Data"

    cal_mat = pd.DataFrame(
        [[10.0, np.nan] + [5.0] * 10],
        index=[2024],
        columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    )
    fig_cal = st_app._make_calendar_heatmap(cal_mat, "Viridis", "Harmonized Hot Spots (1 km cells)")
    assert fig_cal.layout.plot_bgcolor == "#1e293b"

    anom_df = pd.DataFrame(
        {
            "year": [2024, 2024],
            "month": [1, 2],
            "month_name": ["Jan", "Feb"],
            "value": [20.0, np.nan],
            "median": [10.0, 12.0],
            "p25": [5.0, 6.0],
            "p75": [15.0, 14.0],
            "anomaly_pct": [100.0, np.nan],
            "z_score": [1.8, np.nan],
            "robust_z_score": [2.0, np.nan],
            "anomaly_status": ["Critical Anomaly (Robust Z > +1.5)", "No Data"],
        }
    )
    fig_anom = st_app._make_anomaly_bar_chart(anom_df, 2024, "Robust Z-score (0.7413·IQR)")
    assert len(fig_anom.data) == 1

    crit_df = pd.DataFrame(
        {
            "week": [1, 2],
            "week_label": ["W01 (Jan 01)", "W02 (Jan 08)"],
            "p_exceedance": [0.1, 0.75],
            "threshold_p75": [10.0, 10.0],
            "mean_weekly": [2.0, 18.0],
            "max_weekly": [5.0, 40.0],
            "n_years": [10, 10],
            "is_critical": [False, True],
        }
    )
    fig_crit = st_app._make_critical_windows_chart(crit_df)
    assert len(fig_crit.data) == 1

    alpha_df = pd.DataFrame(
        {
            "month": [1, 1],
            "daynight": ["D", "N"],
            "alpha_cells": [1.0, 1.0],
            "alpha_ci_low": [np.nan, np.nan],
            "alpha_ci_high": [np.nan, np.nan],
        }
    )
    fig_alpha = st_app._make_alpha_bar_chart(alpha_df)
    assert len(fig_alpha.data) == 2

