"""Automated tests for real Córdoba FIRMS harmonization, out-of-sample metrics, and API safety."""

from __future__ import annotations

import requests

from src.config import REAL_DIR, REGION_PRESETS, SAMPLE_DIR
from src.data_loader import FIRMSApiClient, _safe_http_error, load_real_cordoba_dataset
from src.harmonizer import HarmonizationPipeline, evaluate_2012_transition


def test_real_cordoba_dataset_bounds_and_size() -> None:
    modis_csv = REAL_DIR / "modis_cordoba_real_2001_2025.csv"
    viirs_csv = REAL_DIR / "viirs_cordoba_real_2012_2025.csv"
    assert modis_csv.exists() and viirs_csv.exists()
    assert modis_csv.stat().st_size < 15 * 1024 * 1024
    assert viirs_csv.stat().st_size < 15 * 1024 * 1024

    m_df, v_df = load_real_cordoba_dataset()
    assert len(m_df) == 21330
    assert len(v_df) == 37294

    west, south, east, north = REGION_PRESETS["cordoba_argentina"]["bbox"]
    for df in (m_df, v_df):
        assert df["longitude"].between(west, east).all()
        assert df["latitude"].between(south, north).all()


def test_real_cordoba_harmonization_and_oos_metrics() -> None:
    m_df, v_df = load_real_cordoba_dataset()
    res = HarmonizationPipeline().run(m_df, v_df)
    tm = res.transition_metrics

    assert tm["has_valid_overlap"] is True
    assert 0.0 < tm["mean_alpha_night"] < tm["mean_alpha_day"] < 1.0
    assert abs(tm["overlap_sensor_residual_bias_pct"]) <= 0.5
    assert abs(tm["harmonized_diff_pct"] - tm["modis_natural_change_pct"]) <= 1.0
    assert tm["overlap_r2"] >= 0.95
    assert tm["deseasonalized_anomaly_r"] >= 0.95
    assert tm["jump_reduction_pct"] > 70.0
    assert -30.0 <= tm["holdout_bias_pct"] <= 30.0
    assert not bool(res.calibration_df["used_fallback"].all())
    assert not ((res.calibration_df["alpha_ci_low"] == 0.1) & (res.calibration_df["alpha_ci_high"] == 0.1)).any()


def test_no_overlap_pre2012_window_flagged() -> None:
    m_df, v_df = load_real_cordoba_dataset()
    pre_m = m_df[m_df["acq_date"] <= "2011-12-31"].copy()
    empty_v = v_df.iloc[0:0].copy()
    res = HarmonizationPipeline().run(pre_m, empty_v)
    assert res.transition_metrics["has_valid_overlap"] is False
    assert bool(res.calibration_df["used_fallback"].all())

    tm = evaluate_2012_transition(
        res.monthly_df, res.calibration_df, {"overlap_r2": 0.0, "overlap_pearson_r": 0.0}
    )
    assert tm["has_valid_overlap"] is False


def test_firms_api_5day_limit_and_secret_redaction() -> None:
    secret_key = "SUPER_SECRET_TEST_KEY_12345"
    req_err = requests.HTTPError(
        f"400 Client Error: Bad Request for url: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{secret_key}/VIIRS_SNPP_NRT/0,0,1,1/7/2025-01-01"
    )
    resp = requests.Response()
    resp.status_code = 400
    req_err.response = resp

    safe_msg = _safe_http_error("FIRMS Area API request failed", req_err)
    assert secret_key not in safe_msg
    assert "HTTP 400" in safe_msg

    client = FIRMSApiClient(map_key=secret_key)
    try:
        client.fetch_area_csv("VIIRS_SNPP_NRT", (-65.0, -32.0, -64.0, -31.0), day_range=7)
        raise AssertionError("Expected ValueError for day_range=7")
    except ValueError as exc:
        assert "between 1 and 5" in str(exc)


def test_synthetic_modis_terra_only_before_aqua_launch() -> None:
    import pandas as pd

    modis_sample = SAMPLE_DIR / "synthetic_modis_cordoba_argentina_2001_2026.csv"
    df = pd.read_csv(modis_sample, usecols=["acq_date", "satellite"])
    pre_aqua = df[df["acq_date"] < "2002-07-04"]
    assert not pre_aqua.empty
    assert set(pre_aqua["satellite"].unique()) == {"Terra"}
