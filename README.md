# AAVAIL Revenue Forecasting – IBM AI Enterprise Workflow Capstone

Forecasts next-30-day revenue for a country (or all) as of any given date.

## Layout
- `app.py` Flask API: `POST /train`, `POST /predict`, `GET /logs`, `GET /logs/<file>`
- `cslib/` ingestion, model, logger, config
- `tests/` + `run_tests.py` unit tests for API, model, logging (isolated via `test=true` dirs / temp data)
- `scripts/` EDA, simulation + monitoring/performance analysis
- `reports/final_report.md`

## Run
```
pip install -r requirements.txt
python run_tests.py
docker build -t aavail-api .
docker run -p 8080:8080 -v $(pwd)/data:/app/data aavail-api
curl -X POST localhost:8080/train -H 'Content-Type: application/json' -d '{}'
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' -d '{"country":"all","year":2019,"month":6,"day":15}'
python scripts/simulate_and_monitor.py --start 2019-01-01 --end 2019-06-30 --country all
```
Put the case-study JSON files in `data/cs-train/`. Retraining = call `/train` (reads every file in the dir), e.g. via cron.
