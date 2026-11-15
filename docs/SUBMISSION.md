# NASA Space Apps Challenge 2026 — Project Page Submission Guide

**Team Name:** VISLUCHKY  
**Team Members:** Yehor Horenych, Felix Romanov, Malieiev Oleksii, Sokolov Volodymyr, Marynets Oleksandr  
**Selected Challenge:** Harmonization of MODIS and VIIRS Hot Spots (NASA & CONAE)  
**Language:** English (required for all Project Page fields and attachments)

---

## 1. Project Name

```text
MODIS & VIIRS Fire Harmonizer
```

---

## 2. High-Level Project Summary

```text
MODIS & VIIRS Fire Harmonizer is an open-source web platform and statistical harmonization pipeline that removes the 2012 cross-sensor discontinuity between NASA MODIS (1 km) and VIIRS (375 m) active fire detections to produce a seamless, climatology-ready Burning Activity Calendar (2001–2026).

Long-term wildfire records cannot be built by concatenating raw MODIS and VIIRS detections: because VIIRS resolves a 1 km MODIS footprint into 4–7 sub-pixels and detects smaller nighttime smoldering fires, switching sensors in March 2012 creates an artificial +235.80% jump in raw detections in our Córdoba, Argentina reference archive (2001–2025). Our 4-stage pipeline normalizes sensor schemas, computes swath pixel area and Fire Radiative Power (FRP) density, grids detections onto a unified 0.01° × 0.01° (~1 km) grid, and applies stratified Day/Night Sum-Ratio calibration α(D/N, m) with Empirical-Bayes shrinkage and 95% bootstrap confidence intervals.

On 58,624 real NASA FIRMS detections over Córdoba Province (2001–2025), our pipeline suppresses the artificial 2012 step-jump by 86.39% (turning +235.80% raw inflation into -32.09%, matching MODIS's own -31.84% physical trend) with R² = 0.9805 overlap agreement and -0.00% overlap residual bias. Users can inspect the validated Córdoba Reference Case or explore any region on Earth via the Live NASA FIRMS Area API or custom multi-year CSV uploads.
```

---

## 3. Link to Project Demo (7-Slide Presentation)

**Important Rule Check:** Official demo presentations are strictly limited to **7 slides maximum (including the title slide)** in **English**. Exceeding 7 slides disqualifies the project from Global Judging.

We generated a **7-slide English PDF presentation** at [`docs/Fire_Harmonizer_Demo_7_Slides.pdf`](Fire_Harmonizer_Demo_7_Slides.pdf) containing:
1. Title slide (Team VISLUCHKY, member, NASA & CONAE challenge, project overview)
2. The Problem (MODIS 1 km vs. VIIRS 375 m discontinuity)
3. How the 4-Stage Pipeline Works (schema unification, swath/FRP density, `0.01°` gridding, stratified Day/Night `α` calibration)
4. Validation on Real NASA Data (Córdoba 2001–2025 proof & metrics)
5. User Experience (1/2): Burning Activity Calendar, Robust Z anomalies & 52-week risk windows
6. User Experience (2/2): Global Live API Explorer, Folium/GIBS map & CSV/JSON export
7. NASA & Space Agency Partner Data (CONAE) + Real-World Impact

**Public Demo URL (copy-paste into the Project Page "Link to Project Demo" field after pushing to GitHub):**
```text
https://github.com/Goshanich/fire-harmonizer/blob/main/docs/Fire_Harmonizer_Demo_7_Slides.pdf
```
*(Alternative: you may also upload `docs/Fire_Harmonizer_Demo_7_Slides.pdf` to Google Drive with "Anyone with the link can view" enabled and paste that public link.)*

---

## 4. Link to Final Project

**Public Live Web Application (Streamlit Community Cloud — no login required):**
```text
https://fire-harmonizer-klk4prhsumtdpkv23tsafg.streamlit.app/
```

**Public Source Code Repository (GitHub):**
```text
https://github.com/Goshanich/fire-harmonizer
```

---

## 5. Detailed Project Description

```text
### What It Does
MODIS & VIIRS Fire Harmonizer reconciles multi-decade active-fire satellite archives from MODIS (Terra & Aqua, 1 km, 2001–present) and VIIRS (Suomi NPP, NOAA-20, NOAA-21, 375 m, 2012–present) into a single, statistically consistent time series and interactive Burning Activity Calendar.

### How It Works (4-Stage Pipeline)
1. Sensor Unification & Confidence Filtering: Normalizes MODIS numeric confidence (0–100, default >= 30) and VIIRS categorical confidence ('l'/'n'/'h', default 'n'/'h') into a unified schema with standardized quality weights.
2. Swath Geometry & FRP Density Normalization: Computes pixel footprint area max(scan × track, 0.05) km² and Fire Radiative Power (FRP) density (MW/km²).
3. Spatiotemporal Gridding (0.01° × 0.01°, ~1 km): Aggregates co-located sub-pixel detections by (grid_lat, grid_lon, acq_date, daynight) into unique active fire cells while preserving raw detection counts and summed FRP.
4. Stratified Sum-Ratio Calibration α(D/N, m): Learns Day (D) and Night (N) VIIRS-to-MODIS scaling ratios from the 2012–2025 concurrent overlap period, blended with monthly Empirical-Bayes shrinkage and 95% bootstrap confidence intervals (1,000 resamples). On the Córdoba real archive, Day α = 0.6564, Night α = 0.3535, and Overall α = 0.5310 (95% CI mean width = 0.0826), reflecting VIIRS's superior sensitivity to smaller nighttime smoldering fires.

### Key Features & Advantages
- Two Operating Modes:
  1. Reference Case — Córdoba, Argentina (2001–2025): Instant access to 58,624 real NASA FIRMS detections (21,330 MODIS C6.1 + 37,294 Suomi NPP VIIRS C2) with full 25-year climatology.
  2. Explore Any Region (Live API / Custom CSV): Free global Bounding Box search using the Live NASA FIRMS Area API (automatic 5-day chunking up to 30 days, exponential backoff retries, response header validation) or custom multi-year CSV uploads from FIRMS Archive Download / Country Yearly Summary.
- NaN-Safe Climatology & Anomaly Detection: Unobserved months remain NaN ('No Data', rendered gray) rather than zero; anomalies use Leave-One-Out Robust Z-scores (0.7413 × IQR); non-stationarity checks warn when multi-year baselines shift; 52-week exceedance profiles (P > P75) identify peak fire-risk weeks (e.g., Week 35 / Aug 27 in Córdoba at 72.0% exceedance).
- Honest Short-Window & Low-Count Behavior: Windows shorter than 2 years or regions with fewer than 30 active 0.01° cells avoid applying foreign climatology and report native VIIRS (α = 1.0) with separate MODIS and VIIRS FRP totals.
- Geospatial Explorer & Reproducible Export: Interactive Folium HeatMap + top 1,000 vector cell markers with NASA GIBS and CONAE OGC WMS overlays, plus 5 downloadable CSV tables and a run_metadata_manifest.json file.

### Technology Stack
- Programming Language: Python 3.12
- Software & Libraries: Streamlit, Pandas, NumPy, SciPy, Plotly, Folium, Requests, PyProj, Pytest (20 automated unit and audit tests)
- Hardware & Deployment: Standard laptop/workstation for development; hosted publicly on Streamlit Community Cloud and GitHub.
```

---

## 6. NASA Data

```text
All satellite fire detections, operational feeds, and map tile layers used in this project come from official NASA Earthdata FIRMS and NASA GIBS resources:

1. MODIS Collection 6.1 Active Fire Detections (MCD14DL / M-C61, 2001–2025):
   - Source: NASA Earthdata FIRMS Archive Download (https://firms.modaps.eosdis.nasa.gov/download/)
   - DOI: 10.5067/FIRMS/MODIS/MCD14DL.NRT.0061
   - Usage: Provides the 2001–2025 1 km baseline active-fire time series (21,330 detections in the Córdoba reference archive) and pre-2012 climatology.

2. VIIRS Suomi NPP Collection 2 Active Fire Detections (VNP14IMG / SV-C2, 2012–2025):
   - Source: NASA Earthdata FIRMS Archive Download (https://firms.modaps.eosdis.nasa.gov/download/)
   - DOI: 10.5067/FIRMS/VIIRS/VNP14IMGTDL.NRT.002
   - Usage: Provides the 375 m high-resolution detections (37,294 detections in Córdoba) used both to calibrate Day/Night α(D/N, m) against concurrent MODIS observations (2012–2025) and to extend the harmonized record.

3. NASA FIRMS Area API & Data Availability API (MODIS_NRT, MODIS_SP, VIIRS_NOAA21_NRT/SP, VIIRS_NOAA20_NRT/SP, VIIRS_SNPP_NRT/SP):
   - Source: NASA FIRMS API (https://firms.modaps.eosdis.nasa.gov/api/area/ and /api/data_availability/)
   - Usage: Powers live, on-demand global queries in "Explore Any Region" mode with automatic 5-day chunking and rate-limit handling.

4. NASA GIBS (Global Imagery Browse Services) OGC WMS:
   - Source: NASA EOSDIS GIBS (https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi)
   - Usage: Renders MODIS/VIIRS thermal anomaly and true-color reflectance WMS overlays inside the interactive Geospatial Explorer map.
```

---

## 7. Space Agency Partner & Other Data

```text
- CONAE (Comisión Nacional de Actividades Espaciales, Argentina — NASA Space Apps 2026 Challenge Partner):
  - Córdoba Province, Argentina (-65.6, -32.6, -63.8, -30.2) serves as our primary real-data validation case, capturing the Sierras de Córdoba and Chaco Serrano winter-spring fire regime.
  - CONAE OGC Web Map Service (WMS) overlays (https://geoservicios.conae.gov.ar/geoserver/wms) are integrated as toggleable reference layers in the Geospatial Explorer map.
- Open-Source Software Ecosystem: Python, Streamlit, Pandas, NumPy, SciPy, Plotly, Folium, PyProj, Requests, and Pytest (MIT License).
```

---

## 8. AI Disclosure

```text
AI coding assistants were used during development for software engineering support, code refactoring, automated test authoring (pytest), documentation drafting, and layout of the 7-slide presentation. All statistical harmonization formulas (sum-ratio calibration, Empirical-Bayes shrinkage, bootstrap confidence intervals, Leave-One-Out Robust Z-scores), NASA FIRMS data processing, and validation runs against the real 2001–2025 Córdoba archive were executed deterministically in Python and verified by the team.
```

---

## 9. Photos & Images (Upload Directly to Project Page)

Upload these 3 PNG screenshots (all under the 10 MB limit) directly in the Project Page image upload section:
1. `screens/01_burning_activity_calendar.png` — Burning Activity Calendar, Robust Z-score anomalies & 52-week Critical Fire Risk Windows
2. `screens/02_harmonization_proof.png` — Raw vs. Harmonized time series & Day/Night α calibration proof (`R² = 0.9805`, `86.39%` jump suppression)
3. `screens/03_geospatial_explorer.png` — Interactive Folium HeatMap, NASA GIBS / CONAE WMS overlays & CSV/JSON export
