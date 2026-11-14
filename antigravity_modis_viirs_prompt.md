# Промпт для Google Antigravity IDE: Реализация проекта «Harmonization of MODIS and VIIRS Hot Spots» (NASA Space Apps Challenge 2026)

Скопируй текст ниже и отправь его агенту в **Google Antigravity IDE** при инициализации рабочего пространства проекта (например, `E:\fire_harmonizer`).

---

## Текст промпта для вставки в Google Antigravity

Ты — Senior Geospatial Data Scientist и Full-Stack Python-разработчик. Мы разрабатываем хакатон-проект для челленджа **[Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)**, предложенного NASA совместно с аргентинским космическим агентством **[CONAE](https://www.argentina.gob.ar/noticias/desafio-nasa-space-apps-challenge-2026-crea-soluciones-para-monitorear-incendios-con-datos)** в рамках NASA Space Apps Challenge 2026.

### 1. Контекст рабочего окружения и правила работы (Workflow Rules)

Соблюдай следующие системные и архитектурные ограничения нашего локального окружения:

* **Дисковое пространство и пути**: Весь проект, виртуальное окружение, скачиваемые CSV-датасеты и кэш размещай строго на диске `E:\` (корневая папка проекта `E:\fire_harmonizer`, кэш пакетов `E:\cache\pip`). Не сохраняй временные файлы и датасеты на диск `C:\`.
* **Аппаратно-программный стек**: Локальная рабочая станция оснащена GPU NVIDIA GeForce RTX 3060 Laptop (CUDA 13.0) и Python 3.11.9. Используй легковесный, быстрый и воспроизводимый стек: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyproj`, `shapely`, `plotly`, `folium` / `streamlit-folium` и `streamlit`.
* **Пошаговый режим разработки (Step-by-Step Mode)**: Не пиши весь код в один монолитный файл. Сначала создай `PROGRESS.md` и `requirements.txt`, после чего реализуй систему по этапам (модулями), проверяя работоспособность каждого скрипта перед переходом к следующему.

### 2. Научно-техническая суть задачи

Спутники дистанционного зондирования Земли фиксируют тепловые аномалии (активные пожары) в среднем и тепловом инфракрасном диапазонах более двух десятилетий ([Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)). Однако исторический архив разделён между двумя типами сенсоров в системе **[NASA Earthdata FIRMS](https://www.earthdata.nasa.gov/data/tools/firms)**, и их сырые записи нельзя сравнивать напрямую ([2026 Challenge Video Explainer](https://www.youtube.com/watch?v=tEPHsiZtuvk)):

| Параметр сравнения | Сенсор MODIS (Terra / Aqua) | Сенсор VIIRS (Suomi NPP / NOAA-20) | Техническое влияние на временной ряд и источник |
| :--- | :--- | :--- | :--- |
| **Период доступности данных** | С ноября 2000 года по настоящее время (выводится из эксплуатации) | С марта 2012 года по настоящее время | При переходе в 2012 году возникает искусственный скачок числа детекций ([VIIRS Global Burned Area Blog](https://www.earthdata.nasa.gov/news/blog/viirs-global-burned-area-product-added-firms)) |
| **Пространственное разрешение** | `1 км × 1 км` в надире (продукт `MCD14DL`) | `375 м × 375 м` в надире (продукт `VNP14IMGTDL_NRT`) | В один пиксель MODIS помещается около 7 пикселей VIIRS ([VIIRS I-Band 375 m Active Fire Data](https://www.earthdata.nasa.gov/data/instruments/viirs/viirs-i-band-375-m-active-fire-data)) |
| **Чувствительность и ночной режим** | Стандартная чувствительность MIR/TIR | Повышенная чувствительность к малым очагам и ночным пожарам | VIIRS фиксирует мелкие и ночные очаги, которые пропускает MODIS ([NASA Earthdata FIRMS](https://www.earthdata.nasa.gov/data/tools/firms)) |
| **Ключевые поля атрибутов** | `latitude`, `longitude`, `brightness`, `scan`, `track`, `acq_date`, `acq_time`, `daynight`, `confidence` (`0–100`), `frp` | `latitude`, `longitude`, `bright_ti4`, `scan`, `track`, `acq_date`, `acq_time`, `daynight`, `confidence` (`l`/`n`/`h`), `frp` | Схемы полей и форматы `confidence` различаются и требуют унификации ([Active Fire Data Attributes for MODIS and VIIRS](https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs)) |

**Цель проекта** состоит из двух связанных частей ([Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)):
1. **Пайплайн гармонизации (`Harmonization Engine`)**: Привести данные MODIS и VIIRS к единой пространственно-временной шкале и откалибровать частоту детекций и мощность теплового излучения (`FRP`), чтобы устранить приборный разрыв 2012 года и получить непрерывный согласованный ряд с 2000 по 2026 год ([2026 Challenge Video Explainer](https://www.youtube.com/watch?v=tEPHsiZtuvk)).
2. **Веб-приложение с календарём активности горения (`Burning Activity Calendar Web App`)**: Построить интерактивный веб-инструмент, где пользователь выбирает регион (`Bounding Box` или координаты `lat/lon`) и диапазон дат, анализирует помесячный тепловой календарь горения (сравнение с исторической медианой и выявление аномалий) и скачивает отфильтрованный гармонизированный датасет в формате **CSV** ([Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)).

### 3. Официальные источники данных и API для интеграции

Используй в коде и документации следующие официальные спецификации и эндпоинты:

* **Основной портал и карта**: **[NASA LANCE FIRMS](https://firms.modaps.eosdis.nasa.gov/)** и интерактивная карта **[FIRMS Global Fire Map](https://firms.modaps.eosdis.nasa.gov/map/)**.
* **Программный доступ по API**: Документация **[FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/)**, руководство **[How to use FIRMS API in Python](https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html)** и получение ключа **[FIRMS Map Key](https://firms.modaps.eosdis.nasa.gov/api/map_key/)**.
  * Эндпоинт выборки по координатам **[FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/)**: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{WEST},{SOUTH},{EAST},{NORTH}/{DAY_RANGE}/{DATE}`
  * Эндпоинт проверки покрытия **[FIRMS Data Availability API](https://firms.modaps.eosdis.nasa.gov/api/data_availability/)**: `https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{MAP_KEY}/{SENSOR}`
* **Архивная выгрузка исторических данных (2000–2026)**: **[FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/)** для пакетной загрузки CSV-файлов по выбранному региону и фильтрации приборных артефактов согласно **[FIRMS FAQ](https://www.earthdata.nasa.gov/data/tools/firms/faq)**.
* **Геосервисы партнёра челленджа (CONAE)**: **[Catálogo de Geoservicios OGC CONAE](https://catalogos.conae.gov.ar/catalogo/catalogoGeoServiciosOGC.html)** (WMS/WFS слои `Focos de Calor MODIS/VIIRS`) и **[Catálogo de Focos de Calor y Áreas Quemadas CONAE](https://catalogos.conae.gov.ar/catalogo/actualizaciones.html)** ([CONAE Space Apps 2026 Announcement](https://www.argentina.gob.ar/noticias/desafio-nasa-space-apps-challenge-2026-crea-soluciones-para-monitorear-incendios-con-datos)).

### 4. Архитектура проекта

Создай следующую структуру каталогов в `E:\fire_harmonizer`:

```text
E:\fire_harmonizer\
├── PROGRESS.md                  # Чек-лист этапов разработки и лог архитектурных решений
├── requirements.txt             # Зависимости проекта
├── README.md                    # Описание алгоритма гармонизации, инструкция запуска и ссылки на NASA/CONAE
├── .env.example                 # Шаблон для FIRMS_MAP_KEY
├── data\
│   ├── raw\                     # Сырые CSV выгрузки MODIS и VIIRS из FIRMS
│   ├── sample\                  # Демо-датасет (2001–2026) для мгновенной работы офлайн без ожидания API
│   └── processed\               # Гармонизированные таблицы и калибровочные коэффициенты
├── src\
│   ├── __init__.py
│   ├── config.py                # Константы, пути на диске E:\, пресеты регионов и палитры
│   ├── data_loader.py           # Клиент NASA FIRMS Area API + загрузчик локальных архивных CSV + генератор реалистичного демо-набора
│   ├── harmonizer.py            # Ядро гармонизации MODIS (1 км) и VIIRS (375 м)
│   └── analytics.py             # Расчёт Burning Activity Calendar, исторических медиан, перцентилей и аномалий
└── app.py                       # Веб-приложение Streamlit с картой, тепловым календарём и экспортом CSV
```

### 5. Требования к алгоритму гармонизации (`src/harmonizer.py`)

Реализуй прозрачный и математически обоснованный пайплайн из 4 шагов на основе спецификации **[Active Fire Data Attributes for MODIS and VIIRS](https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs)**:

1. **Унификация схемы и фильтрация достоверности (`Confidence Filtering`)**:
   * Для MODIS (`MCD14DL` / `MODIS_SP`): числовое поле `confidence` (`0–100`). Отфильтровывай низкодостоверные срабатывания (например, оставляй `confidence >= 30`, с настраиваемым порогом в UI).
   * Для VIIRS (`VIIRS_SNPP_SP` / `VIIRS_NOAA20_NRT`): категориальное поле `confidence` (`l` — low, `n` — nominal, `h` — high). Отбрасывай `l` (low) по умолчанию, конвертируя `n` и `h` в унифицированный вес достоверности ([VIIRS I-Band 375 m Active Fire Data](https://www.earthdata.nasa.gov/data/instruments/viirs/viirs-i-band-375-m-active-fire-data)).
2. **Геометрическая нормализация площади пикселя (`Pixel Area & FRP Normalization`)**:
   * Реальная площадь пикселя на Земле растёт к краям полосы обзора пропорционально `scan × track` (в км²).
   * Рассчитывай для каждой точки эффективную площадь пикселя `pixel_area_km2 = scan * track` и удельную мощность теплового излучения на единицу площади: `frp_density = frp / (scan * track)` (МВт/км²).
3. **Пространственно-временная агрегация в единую сетку (`Spatiotemporal Gridding`)**:
   * Чтобы устранить эффект, когда один пожар даёт 1 точку MODIS (1 км) и кластер из 5–7 точек VIIRS (375 м) ([VIIRS I-Band 375 m Active Fire Data](https://www.earthdata.nasa.gov/data/instruments/viirs/viirs-i-band-375-m-active-fire-data)), проецируй координаты `(latitude, longitude)` на регулярную сетку с шагом `0.01° × 0.01°` (около `1 км × 1 км`, что соответствует нативному разрешению MODIS в надире) с группировкой по дате (`acq_date`) и времени суток пролёта (`daynight`: `D` или `N`).
   * Внутри каждой ячейки сетки `(grid_lat, grid_lon, acq_date, daynight)` агрегируй кластер VIIRS в одну эквивалентную «ячейку очага» (`active_cell_count = 1`), суммируя `frp` и сохраняя исходное число точек `raw_point_count` для сравнения.
4. **Кросс-сенсорная калибровка на периоде перекрытия (2012–2025) (`Overlap Cross-Calibration`)**:
   * На историческом интервале совместной работы MODIS и VIIRS (2012–2025 гг.) для выбранного региона и каждого календарного месяца `m ∈ [1..12]` (и режима `daynight`) вычисляй коэффициент приведения `alpha(m, daynight) = Median(MODIS_active_cells) / Median(VIIRS_active_cells)` (или регрессионный коэффициент с доверительным интервалом 95% методом бутстрапа).
   * Формируй итоговый гармонизированный показатель `harmonized_hotspots`:
     * До 2012 года: данные MODIS (`MODIS_active_cells`).
     * С 2012 года: взвешенная/калиброванная оценка `VIIRS_active_cells * alpha(m, daynight)` (или ансамбль MODIS + откалиброванный VIIRS), что позволяет бесшовно продолжить ряд после отключения MODIS ([Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)).

### 6. Требования к веб-приложению (`app.py`)

Разработай интерактивный дашборд на `Streamlit`, закрывающий все пункты разделов *Objectives* и *Potential Considerations* со страницы **[Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)**:

1. **Боковая панель управления (Sidebar)**:
   * Переключатель источника данных:
     1. Предзагруженные региональные архивы / демо-кейс (работает мгновенно из коробки: например, лесные и степные зоны Украины, провинция Кордова в Аргентине из кейса **[CONAE](https://www.argentina.gob.ar/noticias/desafio-nasa-space-apps-challenge-2026-crea-soluciones-para-monitorear-incendios-con-datos)**, Калифорния).
     2. Загрузка собственных CSV-файлов из **[FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/)**.
     3. Живой запрос через **[FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/)** при вводе `MAP_KEY`.
   * Выбор региона: выбор пресета, ввод координат `Bounding Box` (`min_lon, min_lat, max_lon, max_lat`) или выделение прямоугольника/полигона прямо на интерактивной карте (`Folium Draw`).
   * Фильтры: диапазон лет (например, `2001–2026`), фильтр времени суток (`All / Day / Night`), порог достоверности (`Confidence`), выбор доступной для людей с нарушением цветовосприятия палитры (`Viridis`, `Cividis`, `YlOrRd`).
2. **Вкладка 1 — Календарь активности горения (`Burning Activity Calendar`)**:
   * Главная визуализация: интерактивная тепловая матрица **«Год × Месяц»** (`Plotly Heatmap` с цветовой шкалой `Viridis` / `YlOrRd`), отображающая гармонизированное количество очагов горения и `FRP` с 2001 по 2026 год ([Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)).
   * График сезонного профиля и аномалий: сравнение выбранного года/месяца с **многолетней исторической медианой** и межквартильным размахом (25–75 перцентили), а также столбчатая диаграмма аномалий (`Z-score` или `% отклонения от медианы`), подсвечивающая критические пожароопасные периоды.
   * Карточки ключевых метрик (KPI): пиковый месяц сезона, самый пожароопасный год в истории региона, статус текущего месяца относительно исторической нормы, средний коэффициент калибровки `VIIRS → MODIS`.
3. **Вкладка 2 — Доказательство гармонизации (`Raw vs. Harmonized Comparison`)**:
   * Наглядный график «До и После», который сразу убеждает судей хакатона:
     * Верхний/пунктирный ряд: необработанные (сырые) суммы точек MODIS + VIIRS с резким искусственным скачком в 2012 году после запуска Suomi NPP ([VIIRS I-Band 375 m Active Fire Data](https://www.earthdata.nasa.gov/data/instruments/viirs/viirs-i-band-375-m-active-fire-data)).
     * Нижний/основной ряд: гармонизированный временной ряд после пространственной агрегации `1 км` и применения коэффициентов `alpha(m, daynight)`.
   * Диаграмма рассеяния (`Scatter plot`) корреляции MODIS и VIIRS в период перекрытия (2012–2025) с линией регрессии и формулой пересчёта.
4. **Вкладка 3 — Интерактивная карта очагов и экспорт (`Geospatial Explorer & CSV Export`)**:
   * Карта с отображением тепловых точек/ячеек сетки, разделением слоёв MODIS и VIIRS и возможностью подключения WMS-слоёв **[Catálogo de Geoservicios OGC CONAE](https://catalogos.conae.gov.ar/catalogo/catalogoGeoServiciosOGC.html)** или **[NASA LANCE FIRMS](https://firms.modaps.eosdis.nasa.gov/)**.
   * Кнопки скачивания результатов (**Download CSV**):
     1. Выгрузка гармонизированных точек и ячеек по выбранному региону и датам (`harmonized_hotspots.csv`).
     2. Выгрузка сводной матрицы календаря горения по годам и месяцам (`burning_activity_calendar.csv`).

### 7. Порядок наших действий прямо сейчас

Действуй строго по шагам и запрашивай моё подтверждение после каждого крупного этапа:

1. **Шаг 1 (Инициализация)**: Создай структуру папок проекта, файл `PROGRESS.md` с чек-листом задач и `requirements.txt`.
2. **Шаг 2 (Данные и генератор тестового архива)**: Напиши `src/config.py` и `src/data_loader.py`. Обязательно добавь генератор реалистичного многолетнего датасета (2001–2026 гг.) в `data/sample/` со структурой реальных таблиц **[FIRMS MODIS & VIIRS Attributes](https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs)** (включая сезонные пики, дневные/ночные пролёты и реальное соотношение плотности точек VIIRS 375 м к MODIS 1 км после 2012 года), чтобы мы могли сразу тестировать математику и интерфейс, а также парсер для реальных CSV из **[FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/)** и **[FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/)**.
3. **Шаг 3 (Математическое ядро гармонизации)**: Реализуй `src/harmonizer.py` и `src/analytics.py`, запусти быстрый тест на демо-данных и выведи в консоль метрики сглаживания скачка 2012 года.
4. **Шаг 4 (Веб-интерфейс и экспорт CSV)**: Собери `app.py` на Streamlit, проверь отсутствие ошибок и дай команду для локального запуска.

Начинай с **Шага 1** и **Шага 2**.
