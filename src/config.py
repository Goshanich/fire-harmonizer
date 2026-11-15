"""Configuration constants, disk paths, NASA/CONAE endpoints, and regional presets.

Project root is resolved dynamically so it works both locally on Windows (E:\\)
and inside Linux cloud containers (e.g., Streamlit Community Cloud).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Portable Project Storage Paths (Windows E:\ locally or Linux container)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
REAL_DIR: Path = DATA_DIR / "real"
SAMPLE_DIR: Path = DATA_DIR / "sample"
PROCESSED_DIR: Path = DATA_DIR / "processed"
PIP_CACHE_DIR: Path = (
    Path("E:/cache/pip") if Path("E:/").exists() else (BASE_DIR / ".cache" / "pip")
)


def ensure_project_dirs() -> None:
    """Create required project data directories inside BASE_DIR."""
    for directory in (DATA_DIR, RAW_DIR, REAL_DIR, SAMPLE_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_project_dirs()

# ---------------------------------------------------------------------------
# Official NASA FIRMS & CONAE Data Sources and API Endpoints
# ---------------------------------------------------------------------------
FIRMS_PORTAL_URL: str = "https://firms.modaps.eosdis.nasa.gov/"
FIRMS_MAP_URL: str = "https://firms.modaps.eosdis.nasa.gov/map/"
FIRMS_API_DOCS_URL: str = "https://firms.modaps.eosdis.nasa.gov/api/"
FIRMS_PYTHON_GUIDE_URL: str = (
    "https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html"
)
FIRMS_MAP_KEY_URL: str = "https://firms.modaps.eosdis.nasa.gov/api/map_key/"
FIRMS_ARCHIVE_URL: str = "https://firms.modaps.eosdis.nasa.gov/download/"
FIRMS_COUNTRY_URL: str = "https://firms.modaps.eosdis.nasa.gov/country/"
FIRMS_FAQ_URL: str = "https://www.earthdata.nasa.gov/data/tools/firms/faq"
FIRMS_ATTRIBUTES_URL: str = (
    "https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs"
)

FIRMS_AREA_API_URL: str = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{map_key}/{source}/{west},{south},{east},{north}/{day_range}/{date}"
)
FIRMS_AVAILABILITY_API_URL: str = (
    "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{map_key}/{sensor}"
)

CONAE_CATALOG_OGC_URL: str = (
    "https://catalogos.conae.gov.ar/catalogo/catalogoGeoServiciosOGC.html"
)
CONAE_FOCOS_URL: str = (
    "https://catalogos.conae.gov.ar/catalogo/actualizaciones.html"
)
CONAE_WMS_BASE_URL: str = "https://geoservicios.conae.gov.ar/geoserver/wms"

# ---------------------------------------------------------------------------
# Sensor Definitions, Schemas, and Harmonization Constants
# ---------------------------------------------------------------------------
FIRMS_SOURCES: dict[str, dict[str, Any]] = {
    "MODIS_SP": {
        "sensor": "MODIS",
        "label": "MODIS Standard Processing (MCD14DL)",
        "resolution_m": 1000,
    },
    "MODIS_NRT": {
        "sensor": "MODIS",
        "label": "MODIS Near Real-Time (MCD14DL NRT)",
        "resolution_m": 1000,
    },
    "VIIRS_SNPP_SP": {
        "sensor": "VIIRS",
        "label": "VIIRS Suomi NPP Standard (VNP14IMG)",
        "resolution_m": 375,
    },
    "VIIRS_SNPP_NRT": {
        "sensor": "VIIRS",
        "label": "VIIRS Suomi NPP NRT (VNP14IMGTDL_NRT)",
        "resolution_m": 375,
    },
    "VIIRS_NOAA20_SP": {
        "sensor": "VIIRS",
        "label": "VIIRS NOAA-20 Standard (VJ114IMG)",
        "resolution_m": 375,
    },
    "VIIRS_NOAA20_NRT": {
        "sensor": "VIIRS",
        "label": "VIIRS NOAA-20 NRT (VJ114IMGTDL_NRT)",
        "resolution_m": 375,
    },
    "VIIRS_NOAA21_SP": {
        "sensor": "VIIRS",
        "label": "VIIRS NOAA-21 Standard (VJ214IMG)",
        "resolution_m": 375,
    },
    "VIIRS_NOAA21_NRT": {
        "sensor": "VIIRS",
        "label": "VIIRS NOAA-21 NRT (VJ214IMGTDL_NRT)",
        "resolution_m": 375,
    },
}

MODIS_COLUMNS: list[str] = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
    "type",
]

VIIRS_COLUMNS: list[str] = [
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
    "type",
]

GRID_STEP_DEG: float = 0.01  # ~1 km x 1 km at nadir (native MODIS resolution)
VIIRS_TRANSITION_DATE: str = "2012-03-01"
VIIRS_TRANSITION_YEAR: int = 2012
OVERLAP_YEARS: tuple[int, int] = (2012, 2025)
YEAR_RANGE: tuple[int, int] = (2001, 2026)

DEFAULT_MODIS_MIN_CONFIDENCE: int = 30
DEFAULT_VIIRS_CONFIDENCE: list[str] = ["n", "h"]
VIIRS_CONFIDENCE_WEIGHTS: dict[str, float] = {"l": 0.25, "n": 0.75, "h": 1.0}

# ---------------------------------------------------------------------------
# Colorblind-Safe Palettes
# ---------------------------------------------------------------------------
COLOR_PALETTES: dict[str, dict[str, str]] = {
    "Viridis": {
        "plotly": "Viridis",
        "label": "Viridis (Perceptually Uniform, Colorblind-Safe)",
        "accent": "#21918c",
    },
    "Cividis": {
        "plotly": "Cividis",
        "label": "Cividis (CVD-Optimized Blue-Yellow)",
        "accent": "#3b528b",
    },
    "YlOrRd": {
        "plotly": "YlOrRd",
        "label": "YlOrRd (Thermal Intensity Yellow-Orange-Red)",
        "accent": "#e31a1c",
    },
}

COLOR_SCALE_MODES: list[str] = [
    "Robust (P95 Clamped)",
    "Linear (Min–Max)",
    "Logarithmic (log1p)",
]


def build_firms_map_url(
    bbox: tuple[float, float, float, float],
    zoom: int = 7,
) -> str:
    """Build an official NASA FIRMS interactive Fire Map deep-link centered on ``bbox``."""
    west, south, east, north = bbox
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    return f"{FIRMS_MAP_URL}#d:24hrs;@{center_lon:.2f},{center_lat:.2f},{int(zoom)}z"


# ---------------------------------------------------------------------------
# Bounding Box Validation
# ---------------------------------------------------------------------------


def validate_bbox(
    bbox: tuple[float, float, float, float],
    is_live_api: bool = False,
    max_span_deg: float = 10.0,
    api_days: int = 5,
) -> tuple[bool, str]:
    """Validate geographic bounding box coordinates and Live API span/day limits."""
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return False, "Bounding box must contain exactly 4 coordinates: (west, south, east, north)."

    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False, "Bounding box coordinates must be numeric."

    if any(isinstance(raw_v, bool) or not math.isfinite(val) for raw_v, val in zip(bbox, vals)):
        return False, "Bounding box coordinates must be finite numbers."

    west, south, east, north = vals
    if west < -180.0 or east > 180.0 or west > 180.0 or east < -180.0:
        return False, "Longitude bounds must stay within [-180.0, 180.0] degrees."
    if south < -90.0 or north > 90.0 or south > 90.0 or north < -90.0:
        return False, "Latitude bounds must stay within [-90.0, 90.0] degrees."
    if west >= east:
        return (
            False,
            f"Invalid longitude bounds: Min Lon (West={west:.3f}) must be strictly less than Max Lon (East={east:.3f}); antimeridian-crossing or inverted boxes are not supported.",
        )
    if south >= north:
        return (
            False,
            f"Invalid latitude bounds: Min Lat (South={south:.3f}) must be strictly less than Max Lat (North={north:.3f}).",
        )

    if is_live_api:
        lon_span = east - west
        lat_span = north - south
        if lon_span > max_span_deg or lat_span > max_span_deg:
            return (
                False,
                f"Live NASA FIRMS Area API bounding box cannot exceed {max_span_deg:.1f}° × {max_span_deg:.1f}° (current span: {lon_span:.2f}° × {lat_span:.2f}°).",
            )
        if isinstance(api_days, bool) or int(api_days) < 1 or int(api_days) > 30:
            return False, "Live NASA FIRMS Area API window length must be between 1 and 30 days."

    return True, ""


# ---------------------------------------------------------------------------
# Regional Presets
# ---------------------------------------------------------------------------
REGION_PRESETS: dict[str, dict[str, Any]] = {
    "cordoba_argentina": {
        "name": "Córdoba Province, Argentina (CONAE Case)",
        "name_ru": "Провинция Кордова, Аргентина (кейс CONAE)",
        "description": (
            "Sierras de Córdoba & Chaco Serrano — flagship CONAE fire-monitoring "
            "region with winter-spring biomass burning peaks (Aug–Oct)."
        ),
        "bbox": (-65.6, -32.6, -63.8, -30.2),  # (west, south, east, north)
        "center": (-31.4, -64.7),
        "zoom": 7,
        "peak_months": [8, 9, 10],
        "monthly_weights": [
            0.22, 0.20, 0.25, 0.30, 0.45, 0.70,
            1.15, 2.10, 2.60, 1.95, 0.75, 0.33,
        ],
        "anomalous_years": {
            2003: 1.75,
            2009: 1.65,
            2013: 1.80,
            2020: 2.35,
            2024: 2.10,
        },
        "base_monthly_events": 38,
        "hotspot_centers": [
            (-31.10, -64.55, 0.28, 0.22),
            (-31.65, -64.85, 0.30, 0.25),
            (-32.10, -64.60, 0.25, 0.20),
            (-30.65, -64.30, 0.22, 0.22),
            (-31.40, -65.15, 0.24, 0.20),
        ],
    },
}
