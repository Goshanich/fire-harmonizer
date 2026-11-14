"""Configuration constants, disk paths, NASA/CONAE endpoints, and regional presets.

Project root is resolved dynamically so it works both locally on Windows (E:\\)
and inside Linux cloud containers (e.g., Streamlit Community Cloud).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Portable Project Storage Paths (Windows E:\ locally or Linux container)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
SAMPLE_DIR: Path = DATA_DIR / "sample"
PROCESSED_DIR: Path = DATA_DIR / "processed"
PIP_CACHE_DIR: Path = (
    Path("E:/cache/pip") if Path("E:/").exists() else (BASE_DIR / ".cache" / "pip")
)


def ensure_project_dirs() -> None:
    """Create required project data directories inside BASE_DIR."""
    for directory in (DATA_DIR, RAW_DIR, SAMPLE_DIR, PROCESSED_DIR):
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
    "ukraine_forest_steppe": {
        "name": "Ukraine Forest & Steppe Zones",
        "name_ru": "Лесные и степные зоны Украины",
        "description": (
            "Polissya, Forest-Steppe & Northern Steppe — bimodal fire regime with "
            "spring agricultural/grassland burning (Apr) and late-summer drought peaks (Jul–Sep)."
        ),
        "bbox": (29.5, 46.5, 38.5, 51.8),
        "center": (49.15, 34.0),
        "zoom": 6,
        "peak_months": [4, 7, 8, 9],
        "monthly_weights": [
            0.10, 0.15, 0.85, 1.85, 0.90, 0.65,
            1.35, 2.05, 1.80, 0.85, 0.30, 0.15,
        ],
        "anomalous_years": {
            2007: 1.65,
            2010: 1.85,
            2015: 1.75,
            2020: 2.25,
            2022: 2.40,
        },
        "base_monthly_events": 55,
        "hotspot_centers": [
            (51.20, 30.20, 0.35, 0.55),
            (50.45, 32.80, 0.45, 0.70),
            (49.55, 36.20, 0.55, 0.85),
            (48.35, 35.10, 0.50, 0.80),
            (47.40, 33.40, 0.45, 0.75),
            (48.90, 37.60, 0.40, 0.55),
        ],
    },
    "california_usa": {
        "name": "Northern & Central California, USA",
        "name_ru": "Северная и Центральная Калифорния, США",
        "description": (
            "Sierra Nevada foothills, Sacramento Valley & Coastal Ranges — Mediterranean "
            "fire regime with intense late-summer/autumn wildfire seasons (Jul–Oct)."
        ),
        "bbox": (-123.5, 36.5, -118.5, 41.5),
        "center": (39.0, -121.0),
        "zoom": 6,
        "peak_months": [7, 8, 9, 10],
        "monthly_weights": [
            0.12, 0.15, 0.22, 0.35, 0.55, 1.05,
            1.85, 2.55, 2.40, 1.70, 0.70, 0.20,
        ],
        "anomalous_years": {
            2008: 1.70,
            2018: 2.05,
            2020: 2.50,
            2021: 2.15,
            2024: 1.80,
        },
        "base_monthly_events": 48,
        "hotspot_centers": [
            (40.55, -122.40, 0.45, 0.50),
            (39.75, -121.45, 0.40, 0.45),
            (38.90, -122.65, 0.40, 0.40),
            (37.85, -120.15, 0.45, 0.45),
            (37.10, -119.35, 0.35, 0.40),
        ],
    },
}
