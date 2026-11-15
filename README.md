# 🔥 Harmonization of MODIS and VIIRS Hot Spots (NASA Space Apps Challenge 2026)

Веб-платформа и математический пайплайн для гармонизации исторических рядов активных пожаров **MODIS (1 км)** и **VIIRS (375 м)** с построением непрерывного **Календаря активности горения (Burning Activity Calendar, 2001–2026)**, созданная для челленджа [Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/) от NASA и аргентинского космического агентства [CONAE](https://www.argentina.gob.ar/noticias/desafio-nasa-space-apps-challenge-2026-crea-soluciones-para-monitorear-incendios-con-datos).

---

## 1. Проблема приборного разрыва MODIS vs. VIIRS

Исторический архив тепловых аномалий [NASA Earthdata FIRMS](https://www.earthdata.nasa.gov/data/tools/firms) объединяет два поколения сенсоров с разными физическими характеристиками:

| Параметр | MODIS (Terra / Aqua, `MCD14DL`) | VIIRS (Suomi NPP / NOAA-20, `VNP14IMGTDL`) | Влияние на временной ряд |
| --- | --- | --- | --- |
| **Период данных** | С ноября 2000 г. (выводится из эксплуатации) | С марта 2012 г. по настоящее время | Искусственный скачок детекций в 2012 г. |
| **Разрешение в надире** | 1 км × 1 км | 375 м × 375 м | В 1 пиксель MODIS помещается ~7 пикселей VIIRS |
| **Ночная и малая чувствительность** | Стандартная (MIR/TIR) | Повышенная к малым очагам и ночным пожарам | VIIRS фиксирует больше мелких/ночных пожаров |
| **Поля достоверности** | `confidence`: `0–100` | `confidence`: `'l'` / `'n'` / `'h'` | Несовместимые схемы атрибутов |

---

## 2. Математический алгоритм гармонизации (`src/harmonizer.py`)

Пайплайн выполняет 4 последовательных этапа:

### Шаг 1. Унификация схемы и фильтрация достоверности
- **MODIS**: фильтрация `confidence >= threshold` (по умолчанию `30`), нормированный вес $w = \text{confidence} / 100$.
- **VIIRS**: фильтрация категорий достоверности `confidence ∈ {'n', 'h'}` (отбрасывание низкодостоверных `'l'` по умолчанию), веса `{'l': 0.25, 'n': 0.75, 'h': 1.0}`.

### Шаг 2. Геометрическая нормализация площади пикселя и FRP
Площадь пикселя на краях полосы обзора растёт пропорционально `scan × track`:
$$\text{pixel\_area\_km2} = \max(\text{scan} \times \text{track},\ 0.05)$$
$$\text{frp\_density} = \frac{\text{FRP}}{\text{pixel\_area\_km2}} \quad [\text{МВт/км}^2]$$

### Шаг 3. Пространственно-временная агрегация в сетку $0.01^\circ \times 0.01^\circ$
Координаты проецируются на регулярную сетку шагом $0.01^\circ$ (~1 км × 1 км, нативное разрешение MODIS) с группировкой по `(grid_lat, grid_lon, acq_date, daynight)`:
$$\text{grid\_lat} = \left\lfloor \frac{\text{lat}}{0.01} \right\rfloor \cdot 0.01, \quad \text{grid\_lon} = \left\lfloor \frac{\text{lon}}{0.01} \right\rfloor \cdot 0.01$$
Кластер из 4–7 субпикселей VIIRS внутри одной ячейки 1 км² схлопывается в одну эквивалентную активную ячейку (`active_cell_count = 1`) с суммированием мощности теплового излучения (`frp_sum`) и сохранением числа исходных точек (`raw_point_count`).

### Шаг 4. Кросс-сенсорная калибровка на периоде перекрытия (2012–2025)
Для устранения остаточной разницы чувствительности к мелким и ночным очагам на интервале совместной работы MODIS и VIIRS (2012–2025 гг.) вычисляются помесячные коэффициенты с разделением на дневные (`D`) и ночные (`N`) пролёты:
$$\alpha(m, \text{daynight}) = \frac{\operatorname{Median}\left(\text{MODIS\_active\_cells}_{m, \text{daynight}}\right)}{\operatorname{Median}\left(\text{VIIRS\_active\_cells}_{m, \text{daynight}}\right)}$$
Дополнительно методом бутстрапа (500 итераций) оценивается 95% доверительный интервал и выполняется линейная регрессия (OLS, $R^2$, $r$ Пирсона).

Итоговый гармонизированный ряд:
$$\text{Harmonized}(t) = \begin{cases} \text{MODIS\_active\_cells}(t), & t < \text{2012-03} \\ \sum_{d \in \{D,N\}} \alpha(m, d) \cdot \text{VIIRS\_active\_cells}(t, d), & t \ge \text{2012-03} \quad (\text{режим } \texttt{calibrated\_viirs}) \end{cases}$$
*(Также поддерживается режим ансамбля $\frac{1}{2}(\text{MODIS} + \alpha \cdot \text{VIIRS})$ на годы совместной работы с бесшовным переходом на $\alpha \cdot \text{VIIRS}$ после вывода MODIS из эксплуатации.)*

---

## 3. Результаты верификации (реальный архив NASA FIRMS и синтетический бенчмарк)

### 3.1. Реальный спутниковый архив NASA FIRMS — Провинция Кордова, Аргентина (`2001–2025`, `data/real/`)

Источники: **MODIS Collection 6.1 (`MCD14DL`, `21 330` детекций)** и **Suomi NPP VIIRS Collection 2 (`VNP14IMG`, `37 294` детекций)**, фильтрованные по рамке Кордовы `[-65.6, -32.6, -63.8, -30.2]`.

| Показатель (Real NASA FIRMS Córdoba 2001–2025) | Значение |
| --- | --- |
| **Сырой скачок после 2012 г. (`2001–2011` → `2013–2025`)** | **+224.89%** (`1 014.6` → `3 296.5` точек/год) |
| **После сетки 0.01° без калибровки $\alpha$** | **+23.75%** (`931.5` → `1 152.7` ячеек/год) |
| **Подавление искусственного скачка 2012 г.** | **82.28%** |
| **Коэффициенты калибровки $\alpha$ (`VIIRS → MODIS`)** | `Overall = 0.4049` · **`Day = 0.6426`** · **`Night = 0.1673`** |
| **Остаточный сдвиг сенсоров на перекрытии (`2012–2025`)** | **-9.25%** |
| **Out-of-Sample Holdout Bias (`2012–2018` → `2019–2025`)** | **-19.49%** |
| **Линейное согласие на перекрытии ($R^2$ / $r$ Пирсона)** | **$R^2 = 0.9805$** ($r = 0.9902$) |
| **Корреляция десезонированных аномалий (`deseasonalized_anomaly_r`)** | **`0.9757`** |
| **Пиковый месяц / Пиковый год активности** | `Sep` (медиана `160.0` ячеек) / `2003` (`2 144` ячеек, `224 167.8 MW`) |

### 3.2. Синтетический валидационный бенчмарк (Ground-Truth Recovery Test, `2001–2025`, `data/sample/`)

| Регион | Сырой скачок (2001–2011 → 2013–2025) | После сетки 0.01° без $\alpha$ | Остаточный сдвиг (2012–2025) | Out-of-Sample Holdout (2019–2025) | Подавление скачка 2012 | $R^2$ перекрытия | Deseasonalized $r$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Córdoba, Argentina (CONAE)** | **+916.4%** | +81.3% | **+0.39%** | **-0.77%** | **97.74%** | `0.9917` | `0.9849` |
| **Ukraine Forest & Steppe** | **+821.0%** | +65.3% | **+2.86%** | **-1.65%** | **98.51%** | `0.9895` | `0.9857` |
| **Northern & Central California** | **+849.9%** | +70.4% | **+4.45%** | **+4.85%** | **98.04%** | `0.9921` | `0.9910` |

---

## 4. Структура проекта (`E:\fire_harmonizer`)

```text
E:\fire_harmonizer\
├── app.py                   # Веб-приложение Streamlit (3 вкладки, Folium Draw + WMS, экспорт CSV)
├── requirements.txt         # Зависимости проекта
├── PROGRESS.md              # Чек-лист этапов разработки и лог решений
├── README.md                # Документация и инструкция запуска
├── .env.example             # Шаблон для FIRMS_MAP_KEY
├── .gitignore               # Исключение .env, .venv, кэша и временных CSV
├── data\
│   ├── raw\                 # Кэш сырых выгрузок NASA FIRMS Area API
│   ├── sample\              # Демо-архивы MODIS (2001–2026) и VIIRS (2012–2026)
│   └── processed\           # Гармонизированные матрицы и коэффициенты α
└── src\
    ├── __init__.py
    ├── config.py            # Пути на E:\, официальные эндпоинты NASA/CONAE, пресеты регионов
    ├── data_loader.py       # Клиент NASA FIRMS API, парсер CSV и генератор демо-архива
    ├── harmonizer.py        # 4-этапный пайплайн гармонизации и бутстрап-калибровка
    └── analytics.py         # Burning Activity Calendar, медианы, P25–P75, Z-score, KPI
```

---

## 5. Установка и запуск (строго на диске `E:\`)

### Подготовка окружения и ключа API
1. Установите зависимости с кэшем на диске `E:\`:
   ```powershell
   E:\fire_harmonizer\.venv\Scripts\python.exe -m pip install --cache-dir E:\cache\pip -r E:\fire_harmonizer\requirements.txt
   ```
2. Скопируйте `.env.example` в `.env` и укажите ваш ключ [FIRMS Map Key](https://firms.modaps.eosdis.nasa.gov/api/map_key/):
   ```env
   FIRMS_MAP_KEY=ваш_ключ_firms
   ```

### Запуск CLI-проверки модулей
```powershell
cd E:\fire_harmonizer
E:\fire_harmonizer\.venv\Scripts\python.exe -m src.data_loader
E:\fire_harmonizer\.venv\Scripts\python.exe -m src.analytics
```

### Запуск веб-приложения Streamlit
```powershell
cd E:\fire_harmonizer
E:\fire_harmonizer\.venv\Scripts\python.exe -m streamlit run E:\fire_harmonizer\app.py
```

---

## Limitations & Scientific Scope

1. **Synthetic Benchmark vs. Real Satellite Archives (`data/sample/`)**:
   - The bundled offline preset (**Synthetic Benchmark (2001–2026, Offline)**) is a controlled multi-sensor simulation with known latent fire events designed as a **Ground-Truth Recovery Test** (verifying that the 4-stage pipeline recovers pre-2012 MODIS-equivalent activity and removes the ~7.6× VIIRS sub-pixel inflation).
   - For real empirical research and operational fire management, load official satellite exports via **Upload CSV (FIRMS Archive)** ([NASA FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/)) or query live detections via **Live NASA FIRMS Area API** (chunked in 5-day windows per FIRMS API specification).
2. **Out-of-Sample Generalization & Biome Specificity**:
   - Cross-sensor scaling factors $\alpha(m, \text{daynight})$ depend on fuel structure, flame temperature, and diurnal fire behavior (e.g., nighttime smoldering vs. daytime crown fires).
   - Out-of-sample validation trains $\alpha(m, \text{daynight})$ on `2012–2018` and evaluates on the held-out `2019–2025` period (`holdout_bias_pct` within $\pm 5\%$), alongside deseasonalized monthly anomaly correlation (`deseasonalized_anomaly_r > 0.98`). Custom bounding boxes outside the three benchmark biomes should compute local overlap calibration from multi-year FIRMS archives.
3. **Spatial Gridding Resolution ($0.01^\circ \times 0.01^\circ \approx 1\text{ km}^2$)**:
   - Gridding standardizes VIIRS (`375 m`) clusters onto the native MODIS (`1 km`) nadir scale. While this eliminates artificial multi-pixel counting of a single fire perimeter, sub-kilometer spatial detail within individual `1 km` cells is aggregated into `raw_point_count` and `frp_sum`.
4. **Incomplete Calendar Years & Observational Constraints**:
   - The current calendar year (`2026 YTD`) is excluded from post-2012 annual mean jump calculations so partial-year totals do not bias multi-decadal comparisons.
   - Satellite active-fire detections remain subject to cloud cover, heavy smoke obscuration, canopy interception, and orbital overpass timing (`Terra` only prior to July 2002; `Terra + Aqua` from July 2002 onward; `SNPP / NOAA-20` from March 2012 onward).

---

## 6. Официальные источники NASA FIRMS и CONAE
- [NASA Space Apps Challenge 2026 — Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)
- [NASA LANCE FIRMS Portal](https://firms.modaps.eosdis.nasa.gov/) | [FIRMS Global Fire Map](https://firms.modaps.eosdis.nasa.gov/map/)
- [FIRMS API Documentation](https://firms.modaps.eosdis.nasa.gov/api/) | [How to use FIRMS API in Python](https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html)
- [FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/) | [FIRMS FAQ](https://www.earthdata.nasa.gov/data/tools/firms/faq)
- [Active Fire Data Attributes for MODIS and VIIRS](https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs)
- [VIIRS I-Band 375 m Active Fire Data](https://www.earthdata.nasa.gov/data/instruments/viirs/viirs-i-band-375-m-active-fire-data)
- [Catálogo de Geoservicios OGC CONAE](https://catalogos.conae.gov.ar/catalogo/catalogoGeoServiciosOGC.html) | [Focos de Calor y Áreas Quemadas CONAE](https://catalogos.conae.gov.ar/catalogo/actualizaciones.html)
