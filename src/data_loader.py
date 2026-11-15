"""Data loading, NASA FIRMS API client, CSV normalization, and realistic sample generator.

Supports:
1. Generating reproducible multi-year (2001–2026) synthetic FIRMS MODIS & VIIRS archives
   in E:\\fire_harmonizer\\data\\sample\\ with authentic attribute schemas, seasonal peaks,
   day/night detection asymmetry, and VIIRS 375 m sub-pixel clustering.
2. Loading and parsing real CSV downloads from NASA FIRMS Archive Download or uploaded files.
3. Live querying via NASA FIRMS Area API and Data Availability API with local caching on E:\\.
"""

from __future__ import annotations

import calendar
import os
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import (
    BASE_DIR,
    FIRMS_AREA_API_URL,
    FIRMS_AVAILABILITY_API_URL,
    FIRMS_SOURCES,
    GRID_STEP_DEG,
    MODIS_COLUMNS,
    RAW_DIR,
    REGION_PRESETS,
    SAMPLE_DIR,
    VIIRS_COLUMNS,
    VIIRS_TRANSITION_DATE,
)

load_dotenv(BASE_DIR / ".env")


class FIRMSApiError(RuntimeError):
    """Raised when a NASA FIRMS API request fails or returns an invalid payload."""


def detect_sensor_type(df: pd.DataFrame) -> str:
    """Detect whether a FIRMS dataframe contains MODIS or VIIRS records.

    Returns 'MODIS', 'VIIRS', or 'MIXED'.
    """
    if df.empty:
        if "bright_ti4" in df.columns:
            return "VIIRS"
        if "brightness" in df.columns:
            return "MODIS"
        raise ValueError("Cannot infer sensor type from an empty dataframe without sensor columns.")

    cols = set(df.columns.str.lower())
    has_instrument = "instrument" in cols
    if has_instrument:
        instruments = set(df["instrument"].dropna().astype(str).str.upper().unique())
        has_modis = "MODIS" in instruments
        has_viirs = "VIIRS" in instruments
        if has_modis and has_viirs:
            return "MIXED"
        if has_modis:
            return "MODIS"
        if has_viirs:
            return "VIIRS"

    if "bright_ti4" in cols and "brightness" not in cols:
        return "VIIRS"
    if "brightness" in cols and "bright_ti4" not in cols:
        return "MODIS"
    if "confidence" in cols:
        sample_conf = df["confidence"].dropna().astype(str).str.lower()
        if sample_conf.isin(["l", "n", "h"]).any():
            return "VIIRS"
    return "MODIS"


def _normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize coordinate, date, time, FRP, and daynight columns on a FIRMS dataframe."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required_base = {"latitude", "longitude", "acq_date", "frp"}
    missing = required_base - set(out.columns)
    if missing:
        raise ValueError(f"Missing required FIRMS columns: {sorted(missing)}")

    if out.empty:
        return out

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["frp"] = pd.to_numeric(out["frp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    for col in ("scan", "track"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1.0).clip(lower=0.1)
        else:
            out[col] = 1.0

    out["acq_date"] = pd.to_datetime(out["acq_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["latitude", "longitude", "acq_date"])

    if "acq_time" in out.columns:
        out["acq_time"] = (
            out["acq_time"]
            .fillna("1200")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
            .str.zfill(4)
        )
    else:
        out["acq_time"] = "1200"

    if "daynight" in out.columns:
        out["daynight"] = (
            out["daynight"]
            .fillna("D")
            .astype(str)
            .str.strip()
            .str.upper()
            .where(lambda s: s.isin(["D", "N"]), "D")
        )
    else:
        hours = pd.to_numeric(out["acq_time"].str[:2], errors="coerce").fillna(12)
        out["daynight"] = np.where((hours >= 6) & (hours < 18), "D", "N")

    return out


def split_modis_and_viirs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a normalized or mixed FIRMS dataframe into separate (modis_df, viirs_df)."""
    norm = _normalize_common_columns(df)
    sensor = detect_sensor_type(norm)
    if sensor == "MODIS":
        return norm.reset_index(drop=True), pd.DataFrame(columns=VIIRS_COLUMNS)
    if sensor == "VIIRS":
        return pd.DataFrame(columns=MODIS_COLUMNS), norm.reset_index(drop=True)

    instr = norm["instrument"].astype(str).str.upper()
    modis_df = norm[instr == "MODIS"].reset_index(drop=True)
    viirs_df = norm[instr == "VIIRS"].reset_index(drop=True)
    return modis_df, viirs_df


def load_csv_dataset(
    source: str | Path | Iterable[str | Path],
    sensor_hint: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one or more FIRMS CSV files (path or file-like buffer) and return (modis_df, viirs_df)."""
    sources = [source] if isinstance(source, (str, Path)) or hasattr(source, "read") else list(source)

    modis_frames: list[pd.DataFrame] = []
    viirs_frames: list[pd.DataFrame] = []

    for item in sources:
        if hasattr(item, "seek"):
            item.seek(0)
        raw_df = pd.read_csv(item)
        if raw_df.empty:
            continue

        norm = _normalize_common_columns(raw_df)
        sensor = sensor_hint.upper() if sensor_hint else detect_sensor_type(norm)
        if sensor == "MODIS":
            modis_frames.append(norm)
        elif sensor == "VIIRS":
            viirs_frames.append(norm)
        else:
            m_part, v_part = split_modis_and_viirs(norm)
            if not m_part.empty:
                modis_frames.append(m_part)
            if not v_part.empty:
                viirs_frames.append(v_part)

    modis_df = (
        pd.concat(modis_frames, ignore_index=True)
        if modis_frames
        else pd.DataFrame(columns=MODIS_COLUMNS)
    )
    viirs_df = (
        pd.concat(viirs_frames, ignore_index=True)
        if viirs_frames
        else pd.DataFrame(columns=VIIRS_COLUMNS)
    )
    return modis_df, viirs_df


def filter_by_bbox_and_dates(
    df: pd.DataFrame,
    bbox: Optional[tuple[float, float, float, float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    daynight: str = "All",
) -> pd.DataFrame:
    """Filter a FIRMS dataframe by bounding box (west, south, east, north), dates, and daynight."""
    if df.empty:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    if bbox is not None:
        west, south, east, north = bbox
        mask &= (
            (df["longitude"] >= west)
            & (df["longitude"] <= east)
            & (df["latitude"] >= south)
            & (df["latitude"] <= north)
        )
    if start_date is not None:
        mask &= df["acq_date"] >= str(start_date)
    if end_date is not None:
        mask &= df["acq_date"] <= str(end_date)
    if daynight in ("D", "N"):
        mask &= df["daynight"] == daynight

    return df.loc[mask].reset_index(drop=True)


def _resolve_firms_map_key(map_key: Optional[str] = None) -> str:
    """Resolve FIRMS MAP_KEY from explicit argument, .env/env var, or st.secrets."""
    explicit = (map_key or "").strip()
    if explicit:
        return explicit

    env_key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if env_key and env_key != "your_firms_map_key_here":
        return env_key

    try:
        import streamlit as st

        secret_key = str(st.secrets.get("FIRMS_MAP_KEY", "") or "").strip()
        if secret_key:
            return secret_key
    except Exception:
        pass

    return env_key


class FIRMSApiClient:
    """Client for NASA LANCE FIRMS Area API and Data Availability API."""

    def __init__(
        self,
        map_key: Optional[str] = None,
        cache_dir: Path = RAW_DIR,
        timeout: int = 30,
    ) -> None:
        self.map_key = _resolve_firms_map_key(map_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _require_map_key(self) -> str:
        if not self.map_key or self.map_key == "your_firms_map_key_here":
            raise FIRMSApiError(
                "FIRMS MAP_KEY is missing. Get a free key at "
                "https://firms.modaps.eosdis.nasa.gov/api/map_key/"
            )
        return self.map_key

    def check_data_availability(self, sensor: str = "ALL") -> pd.DataFrame:
        """Query FIRMS Data Availability API for valid date ranges per sensor."""
        key = self._require_map_key()
        url = FIRMS_AVAILABILITY_API_URL.format(map_key=key, sensor=sensor)
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise FIRMSApiError(f"FIRMS Data Availability request failed: {exc}") from exc

        text = resp.text.strip()
        if not text or text.lower().startswith("invalid"):
            raise FIRMSApiError(f"FIRMS API returned an error response: {text[:200]}")

        return pd.read_csv(StringIO(text))

    def fetch_area_csv(
        self,
        source: str,
        bbox: tuple[float, float, float, float],
        day_range: int = 5,
        target_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch active fire records from FIRMS Area API and cache raw CSV on E:\\.

        Parameters
        ----------
        source : str
            One of FIRMS_SOURCES (e.g. 'MODIS_SP', 'VIIRS_SNPP_NRT').
        bbox : tuple
            (west, south, east, north) in decimal degrees.
        day_range : int
            1 to 5 days allowed by FIRMS Area API per request.
        target_date : str, optional
            Starting YYYY-MM-DD date; defaults to today UTC/local date.
        """
        if source not in FIRMS_SOURCES:
            raise ValueError(f"Unsupported FIRMS source '{source}'. Valid: {list(FIRMS_SOURCES)}")
        if not 1 <= int(day_range) <= 5:
            raise ValueError("FIRMS Area API supports day_range between 1 and 5.")

        key = self._require_map_key()
        west, south, east, north = (round(float(v), 4) for v in bbox)
        date_str = target_date or date.today().strftime("%Y-%m-%d")
        cache_name = (
            f"firms_{source}_{west}_{south}_{east}_{north}_d{day_range}_{date_str}.csv"
        )
        cache_path = self.cache_dir / cache_name

        if use_cache and cache_path.exists():
            return _normalize_common_columns(pd.read_csv(cache_path))

        url = FIRMS_AREA_API_URL.format(
            map_key=key,
            source=source,
            west=west,
            south=south,
            east=east,
            north=north,
            day_range=int(day_range),
            date=date_str,
        )
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise FIRMSApiError(f"FIRMS Area API request failed: {exc}") from exc

        text = resp.text.strip()
        if text.lower().startswith("invalid"):
            raise FIRMSApiError(f"FIRMS API error: {text[:200]}")

        cache_path.write_text(text, encoding="utf-8")
        if not text or "\n" not in text:
            cols = MODIS_COLUMNS if FIRMS_SOURCES[source]["sensor"] == "MODIS" else VIIRS_COLUMNS
            return pd.DataFrame(columns=cols)
        return _normalize_common_columns(pd.read_csv(StringIO(text)))

    def fetch_area_range(
        self,
        source: str,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch multi-day range by chunking into 5-day FIRMS Area API windows."""
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
        if end_dt < start_dt:
            raise ValueError("end_date must be >= start_date")

        chunks: list[pd.DataFrame] = []
        cursor = start_dt
        while cursor <= end_dt:
            remaining = (end_dt - cursor).days + 1
            span = min(5, remaining)
            df_chunk = self.fetch_area_csv(
                source=source,
                bbox=bbox,
                day_range=span,
                target_date=cursor.strftime("%Y-%m-%d"),
                use_cache=use_cache,
            )
            if not df_chunk.empty:
                chunks.append(df_chunk)
            cursor += timedelta(days=span)

        if not chunks:
            cols = MODIS_COLUMNS if FIRMS_SOURCES[source]["sensor"] == "MODIS" else VIIRS_COLUMNS
            return pd.DataFrame(columns=cols)
        return pd.concat(chunks, ignore_index=True).drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Realistic Multi-Year FIRMS Sample Dataset Generator (2001–2026)
# ---------------------------------------------------------------------------
def _sample_event_locations(
    rng: np.random.Generator,
    n_events: int,
    bbox: tuple[float, float, float, float],
    hotspot_centers: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic clustered fire locations inside bbox snapped to 0.01 deg grid."""
    west, south, east, north = bbox
    n_clustered = int(round(n_events * 0.80))
    n_uniform = n_events - n_clustered

    lats = np.empty(n_events, dtype=float)
    lons = np.empty(n_events, dtype=float)

    if n_clustered > 0 and hotspot_centers:
        center_idx = rng.integers(0, len(hotspot_centers), size=n_clustered)
        centers = np.asarray(hotspot_centers, dtype=float)
        c_lat = centers[center_idx, 0]
        c_lon = centers[center_idx, 1]
        s_lat = centers[center_idx, 2]
        s_lon = centers[center_idx, 3]
        lats[:n_clustered] = np.clip(rng.normal(c_lat, s_lat), south + 0.015, north - 0.015)
        lons[:n_clustered] = np.clip(rng.normal(c_lon, s_lon), west + 0.015, east - 0.015)
    else:
        n_uniform = n_events
        n_clustered = 0

    if n_uniform > 0:
        lats[n_clustered:] = rng.uniform(south + 0.015, north - 0.015, size=n_uniform)
        lons[n_clustered:] = rng.uniform(west + 0.015, east - 0.015, size=n_uniform)

    cell_lats = np.floor(lats / GRID_STEP_DEG) * GRID_STEP_DEG
    cell_lons = np.floor(lons / GRID_STEP_DEG) * GRID_STEP_DEG
    return cell_lats, cell_lons


def _format_overpass_time(rng: np.random.Generator, daynight: np.ndarray) -> np.ndarray:
    """Generate realistic sun-synchronous afternoon/night overpass HHMM strings."""
    n = len(daynight)
    is_day = daynight == "D"
    hours = np.where(
        is_day,
        rng.integers(12, 16, size=n),
        rng.choice([0, 1, 2, 3, 22, 23], size=n),
    )
    minutes = rng.integers(0, 60, size=n)
    return np.array([f"{h:02d}{m:02d}" for h, m in zip(hours, minutes)], dtype=object)


def generate_regional_sample_data(
    region_key: str,
    start_year: int = 2001,
    end_year: int = 2026,
    seed: int = 42,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate or load cached realistic MODIS (2001–2026) and VIIRS (2012–2026) CSVs.

    Models:
    - Regional monthly seasonality & documented anomalous fire years.
    - MODIS (1 km): misses a larger fraction of small and nighttime fires.
    - VIIRS (375 m, from 2012-03-01): resolves each fire into 4–7 sub-pixels inside the
      same 0.01 deg grid cell, plus additional small/nighttime detections missed by MODIS.
    - Low-confidence thermal anomalies (~10% MODIS conf < 30, ~12% VIIRS conf == 'l')
      so confidence filtering is meaningfully exercised.
    """
    if region_key not in REGION_PRESETS:
        raise ValueError(f"Unknown region_key '{region_key}'. Valid: {list(REGION_PRESETS)}")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    viirs_start_year = max(start_year, 2012)
    modis_path = SAMPLE_DIR / f"modis_{region_key}_{start_year}_{end_year}.csv"
    viirs_path = SAMPLE_DIR / f"viirs_{region_key}_{viirs_start_year}_{end_year}.csv"

    if not force and modis_path.exists() and viirs_path.exists():
        return pd.read_csv(modis_path, dtype={"acq_time": str}), pd.read_csv(
            viirs_path, dtype={"acq_time": str}
        )

    preset = REGION_PRESETS[region_key]
    bbox: tuple[float, float, float, float] = preset["bbox"]
    monthly_weights: list[float] = preset["monthly_weights"]
    anomalous_years: dict[int, float] = preset["anomalous_years"]
    base_events: int = int(preset["base_monthly_events"])
    hotspot_centers = preset["hotspot_centers"]

    region_offset = sum(ord(ch) for ch in region_key)
    rng = np.random.default_rng(seed + region_offset)

    # Generate latent true fire events across all months in [start_year, end_year]
    event_chunks: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        year_mult = anomalous_years.get(year, float(rng.uniform(0.82, 1.18)))
        for month in range(1, 13):
            expected = max(2.0, base_events * monthly_weights[month - 1] * year_mult)
            n_events = int(rng.poisson(expected))
            if n_events == 0:
                continue

            days_in_month = calendar.monthrange(year, month)[1]
            days = rng.integers(1, days_in_month + 1, size=n_events)
            acq_dates = np.array(
                [f"{year:04d}-{month:02d}-{d:02d}" for d in days], dtype=object
            )
            # ~70% daytime, ~30% nighttime true fire activity
            daynight = rng.choice(["D", "N"], size=n_events, p=[0.70, 0.30])
            cell_lats, cell_lons = _sample_event_locations(rng, n_events, bbox, hotspot_centers)
            # Log-normal fire radiative power (MW)
            true_frp = np.round(rng.lognormal(mean=3.35, sigma=0.72, size=n_events), 1)
            event_chunks.append(
                pd.DataFrame(
                    {
                        "cell_lat": cell_lats,
                        "cell_lon": cell_lons,
                        "acq_date": acq_dates,
                        "daynight": daynight,
                        "true_frp": true_frp,
                        "month": month,
                    }
                )
            )

    events = pd.concat(event_chunks, ignore_index=True)
    # Deduplicate same cell/date/daynight so 1 latent event maps cleanly to 1 grid cell
    events = events.drop_duplicates(subset=["cell_lat", "cell_lon", "acq_date", "daynight"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. MODIS Observations (entire period, 1 km resolution)
    # ------------------------------------------------------------------
    is_day = (events["daynight"] == "D").to_numpy()
    # MODIS detects ~88% of daytime fires and ~52% of nighttime fires
    modis_det_prob = np.where(is_day, 0.88, 0.52)
    modis_detected = events[rng.random(len(events)) < modis_det_prob].reset_index(drop=True)

    n_modis_events = len(modis_detected)
    # Occasionally a large fire triggers 2 adjacent sub-detections inside the cell
    modis_mult = rng.choice([1, 2], size=n_modis_events, p=[0.88, 0.12])
    m_idx = np.repeat(np.arange(n_modis_events), modis_mult)
    m_base = modis_detected.iloc[m_idx].reset_index(drop=True)
    n_m = len(m_base)

    m_lat = np.round(m_base["cell_lat"].to_numpy() + rng.uniform(0.0012, 0.0088, size=n_m), 5)
    m_lon = np.round(m_base["cell_lon"].to_numpy() + rng.uniform(0.0012, 0.0088, size=n_m), 5)
    m_scan = np.round(rng.uniform(1.0, 2.35, size=n_m), 2)
    m_track = np.round(rng.uniform(1.0, 1.48, size=n_m), 2)
    m_split = np.repeat(modis_mult, modis_mult)
    m_frp = np.round(
        (m_base["true_frp"].to_numpy() / m_split) * rng.uniform(0.88, 1.12, size=n_m),
        1,
    )
    m_bright = np.round(312.0 + 18.0 * np.log1p(m_frp) + rng.normal(0.0, 4.5, size=n_m), 1)
    m_t31 = np.round(288.0 + np.where(m_base["daynight"] == "D", 10.0, -4.0) + rng.normal(0.0, 3.0, size=n_m), 1)
    m_conf = np.clip(
        np.round(45.0 + 10.0 * np.log1p(m_frp) + rng.normal(0.0, 12.0, size=n_m)),
        30,
        100,
    ).astype(int)

    modis_valid = pd.DataFrame(
        {
            "latitude": m_lat,
            "longitude": m_lon,
            "brightness": m_bright,
            "scan": m_scan,
            "track": m_track,
            "acq_date": m_base["acq_date"].to_numpy(),
            "acq_time": _format_overpass_time(rng, m_base["daynight"].to_numpy()),
            "satellite": rng.choice(["Terra", "Aqua"], size=n_m),
            "instrument": "MODIS",
            "confidence": m_conf,
            "version": "6.1",
            "bright_t31": m_t31,
            "frp": m_frp,
            "daynight": m_base["daynight"].to_numpy(),
            "type": 0,
        }
    )

    # Inject ~9% low-confidence MODIS false alarms (<30) to validate filtering
    n_m_noise = max(10, int(round(len(modis_valid) * 0.09)))
    noise_sample = modis_valid.sample(n=n_m_noise, replace=True, random_state=seed + 1).reset_index(drop=True)
    noise_sample["latitude"] = np.round(
        rng.uniform(bbox[1] + 0.01, bbox[3] - 0.01, size=n_m_noise), 5
    )
    noise_sample["longitude"] = np.round(
        rng.uniform(bbox[0] + 0.01, bbox[2] - 0.01, size=n_m_noise), 5
    )
    noise_sample["confidence"] = rng.integers(5, 29, size=n_m_noise)
    noise_sample["frp"] = np.round(rng.uniform(2.0, 14.0, size=n_m_noise), 1)
    noise_sample["brightness"] = np.round(rng.uniform(301.0, 314.0, size=n_m_noise), 1)

    modis_df = (
        pd.concat([modis_valid, noise_sample], ignore_index=True)
        .sort_values(["acq_date", "acq_time"])
        .reset_index(drop=True)
    )[MODIS_COLUMNS]

    # ------------------------------------------------------------------
    # 2. VIIRS Observations (from 2012-03-01 onward, 375 m resolution)
    # ------------------------------------------------------------------
    viirs_events_base = events[events["acq_date"] >= VIIRS_TRANSITION_DATE].reset_index(drop=True)
    # VIIRS detects ~97% of daytime and ~93% of nighttime fires
    v_is_day = (viirs_events_base["daynight"] == "D").to_numpy()
    viirs_det_prob = np.where(v_is_day, 0.97, 0.93)
    viirs_detected = viirs_events_base[rng.random(len(viirs_events_base)) < viirs_det_prob].reset_index(drop=True)

    # Plus ~22% extra small/nighttime fires that only VIIRS I-band 375 m resolves
    if len(viirs_detected) > 0:
        n_extra_small = max(5, int(round(len(viirs_detected) * 0.22)))
        extra_idx = rng.integers(0, len(viirs_detected), size=n_extra_small)
        extra_small = viirs_detected.iloc[extra_idx].copy().reset_index(drop=True)
        ex_lats, ex_lons = _sample_event_locations(rng, n_extra_small, bbox, hotspot_centers)
        extra_small["cell_lat"] = ex_lats
        extra_small["cell_lon"] = ex_lons
        # Small/night fires skew strongly toward nighttime
        extra_small["daynight"] = rng.choice(["D", "N"], size=n_extra_small, p=[0.28, 0.72])
        extra_small["true_frp"] = np.round(rng.lognormal(mean=2.3, sigma=0.55, size=n_extra_small), 1)
        viirs_all_events = (
            pd.concat([viirs_detected, extra_small], ignore_index=True)
            .drop_duplicates(subset=["cell_lat", "cell_lon", "acq_date", "daynight"])
            .reset_index(drop=True)
        )
    else:
        viirs_all_events = viirs_detected.copy()

    n_v_events = len(viirs_all_events)
    # 1 km MODIS pixel covers ~4 to 7 VIIRS 375 m detections
    cluster_sizes = rng.integers(4, 8, size=n_v_events)
    v_idx = np.repeat(np.arange(n_v_events), cluster_sizes)
    v_base = viirs_all_events.iloc[v_idx].reset_index(drop=True)
    n_v = len(v_base)

    v_lat = np.round(v_base["cell_lat"].to_numpy() + rng.uniform(0.0012, 0.0088, size=n_v), 5)
    v_lon = np.round(v_base["cell_lon"].to_numpy() + rng.uniform(0.0012, 0.0088, size=n_v), 5)
    v_scan = np.round(rng.uniform(0.38, 0.64, size=n_v), 2)
    v_track = np.round(rng.uniform(0.38, 0.52, size=n_v), 2)
    v_split = np.repeat(cluster_sizes, cluster_sizes)
    # Partition event FRP across sub-pixels with slight I-band sensitivity factor
    v_frp = np.round(
        (v_base["true_frp"].to_numpy() / v_split) * rng.uniform(0.92, 1.24, size=n_v),
        1,
    )
    v_bright4 = np.round(315.0 + 19.0 * np.log1p(v_frp) + rng.normal(0.0, 5.0, size=n_v), 1)
    v_bright5 = np.round(287.0 + np.where(v_base["daynight"] == "D", 10.0, -4.0) + rng.normal(0.0, 3.0, size=n_v), 1)
    v_conf = np.where(
        (v_frp > 12.0) | (rng.random(n_v) < 0.48),
        "h",
        "n",
    )
    v_sat = np.where(v_base["acq_date"] >= "2018-01-01", rng.choice(["N", "N20"], size=n_v), "N")

    viirs_valid = pd.DataFrame(
        {
            "latitude": v_lat,
            "longitude": v_lon,
            "bright_ti4": v_bright4,
            "scan": v_scan,
            "track": v_track,
            "acq_date": v_base["acq_date"].to_numpy(),
            "acq_time": _format_overpass_time(rng, v_base["daynight"].to_numpy()),
            "satellite": v_sat,
            "instrument": "VIIRS",
            "confidence": v_conf,
            "version": "2.0NRT",
            "bright_ti5": v_bright5,
            "frp": v_frp,
            "daynight": v_base["daynight"].to_numpy(),
            "type": 0,
        }
    )

    # Inject ~11% low-confidence ('l') VIIRS detections
    if len(viirs_valid) > 0:
        n_v_noise = max(15, int(round(len(viirs_valid) * 0.11)))
        v_noise = viirs_valid.sample(n=n_v_noise, replace=True, random_state=seed + 2).reset_index(drop=True)
        v_noise["latitude"] = np.round(
            rng.uniform(bbox[1] + 0.01, bbox[3] - 0.01, size=n_v_noise), 5
        )
        v_noise["longitude"] = np.round(
            rng.uniform(bbox[0] + 0.01, bbox[2] - 0.01, size=n_v_noise), 5
        )
        v_noise["confidence"] = "l"
        v_noise["frp"] = np.round(rng.uniform(0.8, 5.5, size=n_v_noise), 1)
        v_noise["bright_ti4"] = np.round(rng.uniform(298.0, 315.0, size=n_v_noise), 1)
        viirs_combined = pd.concat([viirs_valid, v_noise], ignore_index=True)
    else:
        viirs_combined = viirs_valid

    viirs_df = (
        viirs_combined
        .sort_values(["acq_date", "acq_time"])
        .reset_index(drop=True)
    )[VIIRS_COLUMNS]

    modis_df.to_csv(modis_path, index=False)
    viirs_df.to_csv(viirs_path, index=False)
    return modis_df, viirs_df


def generate_all_sample_datasets(force: bool = False) -> dict[str, dict[str, object]]:
    """Generate sample datasets for every preset region and return summary stats."""
    summary: dict[str, dict[str, object]] = {}
    for region_key in REGION_PRESETS:
        modis_df, viirs_df = generate_regional_sample_data(region_key=region_key, force=force)
        modis_post2012 = modis_df[modis_df["acq_date"] >= VIIRS_TRANSITION_DATE]
        viirs_post2012 = viirs_df[viirs_df["acq_date"] >= VIIRS_TRANSITION_DATE]
        raw_ratio = (
            len(viirs_post2012) / max(1, len(modis_post2012))
        )
        summary[region_key] = {
            "modis_rows": len(modis_df),
            "viirs_rows": len(viirs_df),
            "modis_span": (modis_df["acq_date"].min(), modis_df["acq_date"].max()),
            "viirs_span": (viirs_df["acq_date"].min(), viirs_df["acq_date"].max()),
            "post2012_raw_viirs_to_modis_ratio": round(raw_ratio, 2),
            "modis_file": str(SAMPLE_DIR / f"modis_{region_key}_2001_2026.csv"),
            "viirs_file": str(SAMPLE_DIR / f"viirs_{region_key}_2012_2026.csv"),
        }
    return summary


if __name__ == "__main__":
    print(f"Generating/verifying FIRMS sample archives under: {SAMPLE_DIR}")
    stats = generate_all_sample_datasets(force=True)
    for key, info in stats.items():
        print(f"\n=== {REGION_PRESETS[key]['name']} ({key}) ===")
        print(f"  MODIS rows : {info['modis_rows']:,}  span={info['modis_span']}")
        print(f"  VIIRS rows : {info['viirs_rows']:,}  span={info['viirs_span']}")
        print(f"  Post-2012 raw VIIRS/MODIS ratio: {info['post2012_raw_viirs_to_modis_ratio']}x")
        print(f"  Files: {info['modis_file']} | {info['viirs_file']}")
