"""Train/predict. A model is fit per country (top N + 'all') to forecast the next-30-day revenue
from features computed at the query date, so predict works for ANY date (15th, month end...)."""
import os, time
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cslib import config, ingest, logger

VERSION = "0.1"
FEATURES = ["rev_7", "rev_14", "rev_30", "rev_60", "rev_90", "purchases_30", "views_30", "month"]

def engineer_features(ts, horizon=config.HORIZON):
    rev = ts["revenue"]
    X = pd.DataFrame(index=ts.index)
    for w in (7, 14, 30, 60, 90):
        X[f"rev_{w}"] = rev.rolling(w, min_periods=1).sum()
    X["purchases_30"] = ts["purchases"].rolling(30, min_periods=1).sum()
    X["views_30"] = ts["views"].rolling(30, min_periods=1).sum()
    X["month"] = ts.index.month
    # target: revenue summed over the NEXT 30 days (t+1 .. t+30)
    y = rev.iloc[::-1].rolling(horizon).sum().iloc[::-1].shift(-1)
    return X[FEATURES], y

def candidates():
    return {"random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0))}

def _rmse(model, X, y):
    cv = TimeSeriesSplit(n_splits=3)
    return -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error").mean()

def train(data_dir=config.DATA_DIR, test=False, cutoff=None, n_countries=10):
    """Compares multiple models + a naive baseline (revenue of last 30 days) via time-series CV."""
    start = time.time()
    df = ingest.load_json_data(data_dir)
    if cutoff is not None:
        df = df[df["date"] <= pd.Timestamp(cutoff)]
    countries = ["all"] + ingest.top_countries(df, n_countries)
    models, report = {}, {}
    for c in countries:
        t0 = time.time()
        X, y = engineer_features(ingest.convert_to_ts(df, c))
        m = y.notna()
        X, y = X[m], y[m]
        if len(X) < 60:
            continue
        scores = {name: _rmse(est, X, y) for name, est in candidates().items()}
        base = float(np.sqrt(np.mean((X["rev_30"] - y) ** 2)))
        best = min(scores, key=scores.get)
        est = candidates()[best].fit(X, y)
        models[c] = {"model": est, "name": best}
        report[c] = {"best_model": best, "rmse": scores[best], "baseline_rmse": base, "all_scores": scores}
        logger.update_train_log(c, best, scores[best], base, time.time() - t0, VERSION,
                                note=f"cutoff={cutoff}", test=test)
    if not models:
        raise ValueError("not enough data to train")
    joblib.dump(models, os.path.join(config.model_dir(test), "models.joblib"))
    return {"runtime": time.time() - start, "report": report}

def load(test=False):
    p = os.path.join(config.model_dir(test), "models.joblib")
    if not os.path.exists(p):
        raise FileNotFoundError("no trained model found; call /train first")
    return joblib.load(p)

def predict(country, year, month, day, data_dir=config.DATA_DIR, test=False):
    """30-day revenue forecast made as of the given date, using only data up to that date."""
    start = time.time()
    models = load(test)
    country = country or "all"
    if country not in models:
        raise ValueError(f"no model for country '{country}'. Available: {sorted(models)}")
    target = pd.Timestamp(year=int(year), month=int(month), day=int(day))
    ts, _ = ingest.fetch_ts(data_dir, country, cutoff=target)
    if target < ts.index.min() + pd.Timedelta(days=1):
        raise ValueError("date is before available data")
    X, _ = engineer_features(ts)
    row = X.loc[[min(target, ts.index.max())]]
    y_pred = float(models[country]["model"].predict(row)[0])
    logger.update_predict_log(country, target.date().isoformat(), y_pred, time.time() - start, VERSION, test)
    return {"country": country, "date": target.date().isoformat(), "y_pred": y_pred, "model": models[country]["name"]}
