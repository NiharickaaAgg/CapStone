# Final Report (fill in with your real numbers)
## 1. Business opportunity
Predict 30-day forward revenue for leadership at arbitrary dates, per country and overall.
## 2. Data & EDA
Describe ingestion, cleaning, top-10 countries, trend/seasonality (see eda_*.png).
## 3. Modeling
Features: rolling revenue windows, purchases, views, month. Candidates: RF, GBM, Ridge. Baseline: last-30-day revenue. Time-series CV RMSE table.
## 4. Deployment
Flask API, Docker, tests, logging, test/prod isolation, retraining via /train.
## 5. Monitoring & performance vs. business metric
Simulation results, model vs baseline plot (performance_*.png), error-vs-revenue analysis.
## 6. Drift, scale, load
Drift: monitor rolling RMSE and input feature distributions; retrain weekly. Scale: per-country models, cache loaded models, move to a DB. Load: gunicorn workers, load test.
## 7. Conclusions & next steps
