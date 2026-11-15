# PROGRESS.md — Harmonization of MODIS and VIIRS Hot Spots (NASA Space Apps Challenge 2026)

🌐 **Live Web App:** [https://fire-harmonizer-klk4prhsumtdpkv23tsafg.streamlit.app/](https://fire-harmonizer-klk4prhsumtdpkv23tsafg.streamlit.app/)

## Контекст проекта
Веб-платформа и математический пайплайн для челленджа **Harmonization of MODIS and VIIRS Hot Spots** (NASA & CONAE, NASA Space Apps Challenge 2026).  
Цель: устранить приборный разрыв 2012 года между **MODIS (1 км × 1 км, с ноября 2000)** и **VIIRS (375 м × 375 м, с марта 2012)**, выполнить калибровку по пролётам Day/Night и построить климатологически корректный **Burning Activity Calendar**.

---

## Чек-лист этапов разработки и технического аудита

### Шаг 1–2 — Инфраструктура, конфигурация и надёжный слой данных (`src/config.py`, `src/data_loader.py`)
- [x] Настроен изолированный venv (`.venv`), `LICENSE` (MIT + официальные DOI NASA FIRMS и атрибуция CONAE), `.gitignore` (защита `.env` и временного кэша `data/raw/`).
- [x] Реализован клиент NASA FIRMS Area API & Data Availability API (`src/data_loader.py`):
  - Чанкинг по **5 дням** (`span = min(5, remaining)`, строгий лимит `1 <= day_range <= 5` по актуальной спецификации NASA FIRMS).
  - Маскировка `MAP_KEY` в текстах ошибок (`_safe_http_error`), экспоненциальный backoff при rate limit, валидация заголовка CSV перед записью в кэш (удаление битых/HTML/rate-limit ответов) и трекинг сбойных 5-дневных окон (`allow_partial=True` + `last_failed_windows`).
  - Парсер пользовательских CSV (`load_csv_dataset_with_report`): автоопределение разделителя (`,` / `;`), ISO `%Y-%m-%d` + опциональный режим `dayfirst`, диагностика отброшенных строк и географического охвата.

### Шаг 3 — Математическое ядро гармонизации (`src/harmonizer.py`, `src/analytics.py`)
- [x] 4-этапный пайплайн (`unify_and_filter_sensors` → `normalize_pixel_geometry` → `grid_spatiotemporal_clusters` на `0.01° × 0.01°` → `compute_overlap_calibration` и сшивка ряда):
  - Переход на **sum-ratio калибровку** $\alpha(\text{D/N}, m)$ с empirical-Bayes усадкой (`w = min(0.35, n/(n+12))`) и точной нормировкой сумм на пролёт (`D`/`N`), дающий остаточный сдвиг на перекрытии `~0.00%`.
  - Парный бутстрап (paired month resampling) для 95% ДИ без вырожденных `[0.1, 0.1]` или хардкодных констант `0.85/0.36`; при нехватке данных — `NaN` ДИ и `calibration_status = "no_data"`.
  - Сравнение перехода на полных годах **`2003–2011` vs `2013–2025`** (исключение неполных лет и пре-Aqua 2001–2002) с отчётом `modis_natural_change_pct`, `holdout_bias_pct` (`2012–2018 -> 2019–2025`) и `deseasonalized_anomaly_r`.
- [x] Аналитика календаря (`src/analytics.py`):
  - Ненаблюдаемые месяцы помечаются `data_status = "no_data"` и остаются `NaN` (серые ячейки в тепловой карте, исключаются из расчёта климатологии).
  - **Leave-One-Out (LOO) Robust Z-score (`0.7413 · IQR`)**, статус `"Insufficient Data (<8 yrs)"` при короткой выборке, проверка нестационарности (`check_series_nonstationarity`) и 52-недельные критические окна пожарного риска (`compute_critical_fire_windows`, `P > P75`).

### Шаг 4 — Двухрежимное веб-приложение Streamlit (`app.py`)
- [x] Режим **Reference Case — Córdoba, Argentina (2001–2025)** (`data/real/`) + 3 биома **Synthetic Validation Benchmark** (`data/sample/synthetic_*.csv`).
- [x] Режим **Explore Any Region (Live API / Custom CSV / GeoJSON AOI)**: быстрые локации (`QUICK_LOCATIONS`), загрузка полигонов `.geojson/.json`, пошаговый гайд по FIRMS Archive Download, диагностика CSV, предупреждения об отсутствии калибровки и сбоях чанков Live API.
- [x] Обновлённые графики (серая маска `No Data`, LOO Robust Z, 52-недельные критические окна, пояснение *How to Read This Calendar*), честные метрики валидации на Tab 2 и экспорт `run_metadata_manifest.json` на Tab 3.

### Шаг 5–6 — Автотесты и финальная верификация (`tests/`, `python -m src.analytics`)
- [x] Набор автотестов (`tests/test_custom_global.py`, `tests/test_harmonizer_real.py`, `tests/test_audit_compliance.py`) — все тесты пройдены (`pytest -q`).

---

## Точные наблюдаемые результаты верификации (`python -m src.analytics`)

### 1. Реальный спутниковый архив NASA FIRMS: Провинция Кордова (`2001–2025`, `data/real/`)
- **Объём:** MODIS C6.1 (`MCD14DL`) = `21,330` детекций, Suomi NPP VIIRS C2 (`VNP14IMG`) = `37,294` детекций (`[-65.6, -32.6, -63.8, -30.2]`).
- **Переход (`2003–2011` vs `2013–2025`):**
  - Сырой скачок точек: **`+235.80%`** (`981.7 -> 3,296.5` точек/год)
  - Без калибровки α (сетка `0.01°`): **`+27.86%`** (`901.6 -> 1,152.7` ячеек/год)
  - Гармонизированный ряд: **`-32.09%`** (`901.6 -> 612.3`) vs физический тренд по одному MODIS (**`modis_natural_change_pct = -31.84%`**)
  - Подавление искусственного скачка: **`86.39%`**
  - Остаточный сдвиг на перекрытии (`2012-03..2025`): **`-0.00%`** | Out-of-Sample Holdout (`2019–2025`): **`-12.99%`**
  - Линейное согласие: **$R^2 = 0.9805$**, $r = 0.9902$, Десезонированная корреляция аномалий $r = 0.9893$
  - Sum-Ratio $\alpha$: **`Overall = 0.5310`** · **`Day = 0.6564`** · **`Night = 0.3535`** (средняя ширина 95% ДИ = **`0.0826`**)
  - Пиковый месяц / год / недельное окно: `Sep` (медиана `160.0`) / `2003` (`2,144.0`) / **`W35 (Aug 27)`** (`72.0%` лет > `P75 = 9.7`)

### 2. Синтетические бенчмарки самосогласованности (`data/sample/synthetic_*.csv`)
- **Córdoba, Argentina (`cordoba_argentina`):** сырой скачок `+898.3%`, сетка без α `+78.4%`, гармонизированный сдвиг `+18.13%` vs MODIS `+18.31%`, перекрытие `+0.00%`, Holdout `+0.76%`, подавление скачка `97.98%`, $R^2 = 0.9917$, Deseasonalized $r = 0.9864$, Sum-Ratio α: Overall `0.6622` (D `0.8409`, N `0.3591`).
- **Ukraine Forest & Steppe (`ukraine_forest_steppe`):** сырой скачок `+794.6%`, сетка без α `+60.4%`, гармонизированный сдвиг `+5.77%` vs MODIS `+5.86%`, перекрытие `+0.00%`, Holdout `-2.53%`, подавление скачка `99.27%`, $R^2 = 0.9895$, Deseasonalized $r = 0.9862$, Sum-Ratio α: Overall `0.6594` (D `0.8405`, N `0.3554`).
- **Northern & Central California (`california_usa`):** сырой скачок `+825.3%`, сетка без α `+66.0%`, гармонизированный сдвиг `+8.77%` vs MODIS `+8.82%`, перекрытие `-0.00%`, Holdout `+3.34%`, подавление скачка `98.94%`, $R^2 = 0.9921$, Deseasonalized $r = 0.9917$, Sum-Ratio α: Overall `0.6554` (D `0.8329`, N `0.3554`).
