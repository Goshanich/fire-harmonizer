# PROGRESS.md — Harmonization of MODIS and VIIRS Hot Spots (NASA Space Apps Challenge 2026)

## Контекст проекта
Проект для челленджа **Harmonization of MODIS and VIIRS Hot Spots** (NASA & CONAE, NASA Space Apps Challenge 2026).
Цель: устранить приборный разрыв 2012 года между сенсорами MODIS (1 км × 1 км, с ноября 2000) и VIIRS (375 м × 375 м, с марта 2012), откалибровать частоту детекций и мощность теплового излучения (FRP) и создать веб-приложение с календарём активности горения (**Burning Activity Calendar**).

---

## Чек-лист этапов разработки

### Шаг 1 — Инициализация проекта
- [x] Создать структуру каталогов на диске `E:\fire_harmonizer` (`data/raw`, `data/sample`, `data/processed`, `src`).
- [x] Настроить кэш пакетов `E:\cache\pip` и временную директорию `E:\fire_harmonizer\.tmp` (без использования `C:\`).
- [x] Создать `requirements.txt`, `.env.example`, `src/__init__.py` и `PROGRESS.md`.
- [x] Проверить и установить зависимости в виртуальном окружении `E:\fire_harmonizer\.venv` (Python 3.11.9).

### Шаг 2 — Конфигурация и слой данных (`src/config.py`, `src/data_loader.py`)
- [x] Реализовать `src/config.py`: пути строго на `E:\`, официальные эндпоинты NASA FIRMS и CONAE OGC WMS/WFS, пресеты регионов (`cordoba_argentina`, `ukraine_forest_steppe`, `california_usa`), палитры (`Viridis`, `Cividis`, `YlOrRd`), параметры сенсоров.
- [x] Реализовать `src/data_loader.py`:
  - Генератор реалистичного многолетнего демо-архива (2001–2026) в `data/sample/` со структурой таблиц FIRMS (`MCD14DL` и `VNP14IMGTDL_NRT`), сезонными пиками, дневными/ночными пролётами (`D`/`N`) и кластерной плотностью VIIRS (4–7 субпикселей на 1 ячейку MODIS после марта 2012).
  - Парсер и нормализатор локальных/пользовательских CSV из FIRMS Archive Download.
  - Клиент NASA FIRMS Area API и Data Availability API с кэшированием на `E:\fire_harmonizer\data\raw`.
- [x] Проверить генерацию и загрузку тестовых датасетов через CLI (`python -m src.data_loader`).

### Шаг 3 — Математическое ядро гармонизации и аналитика (`src/harmonizer.py`, `src/analytics.py`)
- [x] `src/harmonizer.py`:
  1. Унификация схемы и фильтрация достоверности (MODIS `confidence >= threshold`, VIIRS `confidence in ['n', 'h']`).
  2. Геометрическая нормализация площади пикселя (`pixel_area_km2 = scan * track`, `frp_density = frp / (scan * track)`).
  3. Пространственно-временная агрегация в регулярную сетку `0.01° × 0.01°` (~1 км × 1 км) по `(grid_lat, grid_lon, acq_date, daynight)`.
  4. Кросс-сенсорная калибровка на периоде перекрытия (2012–2025): коэффициенты `alpha(m, daynight)` с 95% доверительным интервалом (бутстрап) и сшивка непрерывного ряда 2001–2026.
- [x] `src/analytics.py`: матрица Burning Activity Calendar (Год × Месяц), исторические медианы, 25–75 перцентили, аномалии (Z-score, % отклонения), KPI-метрики и метрики сглаживания скачка 2012 года.
- [x] Проверка авторизации `FIRMSApiClient.check_data_availability("ALL")` через `.env` (`FIRMS_MAP_KEY`) и экспорт обработанных таблиц в `data/processed/`.

### Шаг 4 — Веб-приложение Streamlit и документация (`app.py`, `README.md`)
- [x] Собрать интерактивный дашборд `app.py` (Sidebar + 3 вкладки: Burning Activity Calendar, Доказательство гармонизации Raw vs Harmonized, Интерактивная карта очагов со слоями NASA/CONAE и экспортом CSV).
- [x] Создать `README.md` с описанием алгоритма, ссылками на NASA FIRMS / CONAE и инструкцией запуска.
- [x] Провести сквозную проверку приложения и экспорта CSV.

---

## Лог архитектурных решений
- **Хранение данных и кэша**: Все данные, виртуальное окружение (при создании), временные файлы и кэш pip (`E:\cache\pip`) изолированы на диске `E:\` для защиты системного диска `C:\`.
- **Единая пространственно-временная сетка**: Шаг `0.01° × 0.01°` (~1.11 км на экваторе) соответствует нативному надирному разрешению MODIS (1 км), устраняя искусственное завышение числа срабатываний VIIRS (375 м) за счёт агрегации субпиксельных кластеров в единую ячейку очага (`active_cell_count = 1`) при сохранении суммарной мощности `frp` и исходного числа детекций `raw_point_count`.
- **Учёт ночной и мелкоочаговой чувствительности VIIRS**: Даже после агрегации на сетку 1 км сенсор VIIRS регистрирует больше малых и ночных пожаров, поэтому применяется помесячная и суточная калибровка `alpha(m, daynight)` на периоде перекрытия 2012–2025 гг.
- **Официальные источники и API**:
  - NASA FIRMS Area API: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{WEST},{SOUTH},{EAST},{NORTH}/{DAY_RANGE}/{DATE}`
  - NASA FIRMS Data Availability API: `https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{MAP_KEY}/{SENSOR}`
  - CONAE OGC WMS/WFS Geoservices: `https://catalogos.conae.gov.ar/catalogo/catalogoGeoServiciosOGC.html`
- **Результаты верификации Шага 3 (демо-архив 2001–2026)**:
  - Córdoba, Argentina: искусственный скачок сырых точек `+892.2%` сглажен на `97.92%`, остаточный сенсорный сдвиг на перекрытии (2012–2025): `+0.95%`, `R² = 0.9895` (`alpha Day = 0.8211`, `Night = 0.3579`).
  - Ukraine Forest & Steppe: скачок `+807.9%` сглажен на `98.85%`, остаточный сдвиг: `+1.48%`, `R² = 0.9898` (`alpha Day = 0.8815`, `Night = 0.3569`).
  - California, USA: скачок `+835.8%` сглажен на `98.30%`, остаточный сдвиг: `+4.05%`, `R² = 0.9912` (`alpha Day = 0.8761`, `Night = 0.3667`).
