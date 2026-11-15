"""Unit tests for Explore Any Region mode, BBox validation, and in-window calibration."""

from __future__ import annotations

import pandas as pd

from src.config import FIRMS_SOURCES, QUICK_LOCATIONS, validate_bbox
from src.harmonizer import (
    HarmonizationPipeline,
    build_daily_sensor_breakdown,
    build_in_window_calibration,
    grid_spatiotemporal_clusters,
    normalize_pixel_geometry,
    unify_and_filter_sensors,
)


def _make_viirs_rows(n: int, acq_dates: list[str], daynight: str = "D", frp: float = 12.0) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        rows.append(
            {
                "latitude": 51.0 + idx * 0.02,
                "longitude": -116.5 + (idx % 5) * 0.02,
                "bright_ti4": 330.0,
                "scan": 0.4,
                "track": 0.4,
                "acq_date": acq_dates[idx % len(acq_dates)],
                "acq_time": "1900",
                "satellite": "N20",
                "instrument": "VIIRS",
                "confidence": "n",
                "version": "2.0NRT",
                "bright_ti5": 290.0,
                "frp": frp,
                "daynight": daynight,
                "type": 0,
            }
        )
    return pd.DataFrame(rows)


def _make_modis_rows(n: int, acq_dates: list[str], daynight: str = "D", frp: float = 25.0) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        rows.append(
            {
                "latitude": 51.0 + idx * 0.02,
                "longitude": -116.5 + (idx % 5) * 0.02,
                "brightness": 325.0,
                "scan": 1.0,
                "track": 1.0,
                "acq_date": acq_dates[idx % len(acq_dates)],
                "acq_time": "1845",
                "satellite": "Aqua",
                "instrument": "MODIS",
                "confidence": 75,
                "version": "6.1NRT",
                "bright_t31": 292.0,
                "frp": frp,
                "daynight": daynight,
                "type": 0,
            }
        )
    return pd.DataFrame(rows)


def test_viirs_only_window_uses_native_alpha_1() -> None:
    empty_modis = pd.DataFrame()
    viirs_df = _make_viirs_rows(45, ["2026-10-29", "2026-10-30", "2026-10-31"], frp=15.0)

    unified = unify_and_filter_sensors(empty_modis, viirs_df)
    gridded = grid_spatiotemporal_clusters(normalize_pixel_geometry(unified))
    cal_df, meta = build_in_window_calibration(gridded)

    assert len(cal_df) == 24
    assert set(cal_df["month"]) == set(range(1, 13))
    assert (cal_df["alpha_cells"] == 1.0).all()
    assert bool(cal_df["used_fallback"].all()) is True
    assert meta["used_fallback"] is True
    assert meta["low_sample_warning"] is True
    assert meta["mode_label"] == "VIIRS native (not harmonized to MODIS)"

    result = HarmonizationPipeline().run(
        empty_modis,
        viirs_df,
        calibration_override=cal_df,
    )
    assert (result.calibration_df["alpha_cells"] == 1.0).all()
    assert bool(result.calibration_df["used_fallback"].all()) is True
    assert abs(result.monthly_df["harmonized_hotspots"].sum() - 45.0) < 1e-6


def test_two_month_window_oct_nov_has_no_missing_months() -> None:
    oct_dates = ["2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31"]
    nov_dates = ["2026-11-01", "2026-11-02", "2026-11-03"]
    modis_df = pd.concat(
        [
            _make_modis_rows(20, oct_dates, daynight="D", frp=30.0),
            _make_modis_rows(20, nov_dates, daynight="D", frp=30.0),
        ],
        ignore_index=True,
    )
    viirs_df = pd.concat(
        [
            _make_viirs_rows(40, oct_dates, daynight="D", frp=10.0),
            _make_viirs_rows(40, nov_dates, daynight="D", frp=10.0),
        ],
        ignore_index=True,
    )

    unified = unify_and_filter_sensors(modis_df, viirs_df)
    gridded = grid_spatiotemporal_clusters(normalize_pixel_geometry(unified))
    cal_df, meta = build_in_window_calibration(gridded)

    assert len(cal_df) == 24
    assert set(cal_df["month"]) == set(range(1, 13))
    assert meta["used_fallback"] is False
    assert meta["low_sample_warning"] is False
    assert meta["n_modis_cells"] >= 35
    assert meta["n_viirs_cells"] >= 70
    assert (cal_df["alpha_cells"] < 1.0).all()

    result = HarmonizationPipeline().run(
        modis_df,
        viirs_df,
        calibration_override=cal_df,
    )
    oct_cells = result.harmonized_cells_df[result.harmonized_cells_df["month"] == 10]
    nov_cells = result.harmonized_cells_df[result.harmonized_cells_df["month"] == 11]
    assert not oct_cells.empty and not nov_cells.empty
    assert (oct_cells["alpha_cells"] < 1.0).all()
    assert (nov_cells["alpha_cells"] < 1.0).all()
    assert not bool(result.calibration_df["used_fallback"].any())


def test_bbox_validation_rules() -> None:
    banff_bbox = QUICK_LOCATIONS["Banff National Park, Canada"]
    assert banff_bbox == (-117.317, 50.705, -115.165, 52.271)

    ok_banff, err_banff = validate_bbox(banff_bbox, is_live_api=True, max_span_deg=10.0, api_days=5)
    assert ok_banff is True and err_banff == ""

    # Reject west >= east (inverted and antimeridian crossing)
    ok_inv_lon, _ = validate_bbox((-10.0, 10.0, -20.0, 20.0))
    ok_antimeridian, _ = validate_bbox((170.0, -10.0, -170.0, 10.0))
    assert ok_inv_lon is False
    assert ok_antimeridian is False

    # Reject south >= north
    ok_inv_lat, _ = validate_bbox((-65.0, -30.0, -63.0, -32.0))
    assert ok_inv_lat is False

    # Reject out-of-range coordinates
    assert validate_bbox((-181.0, 0.0, -175.0, 5.0))[0] is False
    assert validate_bbox((175.0, 0.0, 181.0, 5.0))[0] is False
    assert validate_bbox((0.0, -91.0, 5.0, 0.0))[0] is False
    assert validate_bbox((0.0, 0.0, 5.0, 91.0))[0] is False
    assert validate_bbox((float("nan"), 0.0, 5.0, 5.0))[0] is False

    # Live API span cap (> 10 deg) and api_days cap (> 30 or < 1)
    assert validate_bbox((-10.0, 0.0, 2.0, 5.0), is_live_api=True, max_span_deg=10.0)[0] is False
    assert validate_bbox((0.0, -6.0, 5.0, 6.0), is_live_api=True, max_span_deg=10.0)[0] is False
    assert validate_bbox(banff_bbox, is_live_api=True, max_span_deg=10.0, api_days=31)[0] is False
    assert validate_bbox(banff_bbox, is_live_api=True, max_span_deg=10.0, api_days=0)[0] is False


def test_separate_sensor_frp_no_double_counting() -> None:
    dates = ["2026-10-28", "2026-10-29", "2026-10-30"]
    modis_df = _make_modis_rows(36, dates, daynight="D", frp=25.0)  # sum = 900.0 MW
    viirs_df = _make_viirs_rows(72, dates, daynight="D", frp=10.0)  # sum = 720.0 MW

    unified = unify_and_filter_sensors(modis_df, viirs_df)
    gridded = grid_spatiotemporal_clusters(normalize_pixel_geometry(unified))
    cal_df, meta = build_in_window_calibration(gridded)

    assert abs(meta["modis_frp_total"] - 900.0) < 1e-6
    assert abs(meta["viirs_frp_total"] - 720.0) < 1e-6

    result = HarmonizationPipeline(combination_mode="calibrated_viirs").run(
        modis_df, viirs_df, calibration_override=cal_df
    )
    daily = build_daily_sensor_breakdown(result.gridded_df, result.harmonized_cells_df)

    assert "raw_total_frp" not in daily.columns
    assert abs(daily["modis_frp"].sum() - 900.0) < 1e-6
    assert abs(daily["viirs_frp"].sum() - 720.0) < 1e-6
    # Harmonized FRP matches MODIS-equivalent (900.0 MW), never summed raw 1620.0 MW
    assert abs(daily["harmonized_frp"].sum() - 900.0) < 1.0

    # Ensemble mode also must not double-count daily FRP or cells
    res_ens = HarmonizationPipeline(combination_mode="ensemble").run(
        modis_df, viirs_df, calibration_override=cal_df
    )
    daily_ens = build_daily_sensor_breakdown(res_ens.gridded_df, res_ens.harmonized_cells_df)
    assert abs(daily_ens["harmonized_frp"].sum() - 900.0) < 1.0


def test_viirs_noaa21_sources_configured() -> None:
    assert "VIIRS_NOAA21_NRT" in FIRMS_SOURCES
    assert "VIIRS_NOAA21_SP" in FIRMS_SOURCES
    assert FIRMS_SOURCES["VIIRS_NOAA21_NRT"]["sensor"] == "VIIRS"
    assert FIRMS_SOURCES["VIIRS_NOAA21_SP"]["sensor"] == "VIIRS"

