"""Post-production analysis: replay history day by day (retrain weekly with data up to that day only),
log predictions, compare to actuals AND the baseline, and plot.
Usage: python scripts/simulate_and_monitor.py --data data/cs-train --start 2019-01-01 --end 2019-06-30 --country all
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cslib import model, ingest

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/cs-train"); p.add_argument("--start", required=True)
p.add_argument("--end", required=True); p.add_argument("--country", default="all")
p.add_argument("--retrain_days", type=int, default=7); p.add_argument("--out", default="reports")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

ts, _ = ingest.fetch_ts(a.data, a.country)
actual = ts["revenue"].iloc[::-1].rolling(30).sum().iloc[::-1].shift(-1)   # known future 30d revenue
rows = []
for i, d in enumerate(pd.date_range(a.start, a.end)):
    if i % a.retrain_days == 0:
        model.train(data_dir=a.data, test=True, cutoff=d)                  # weekly 're-train'
    r = model.predict(a.country, d.year, d.month, d.day, data_dir=a.data, test=True)
    base = ts["revenue"].loc[:d].tail(30).sum()                            # baseline: last 30d revenue
    rows.append({"date": d, "pred": r["y_pred"], "baseline": base, "actual": actual.get(d, np.nan)})
res = pd.DataFrame(rows).dropna().set_index("date")
res.to_csv(os.path.join(a.out, f"simulation_{a.country}.csv"))

rmse = lambda c: float(np.sqrt(((res[c] - res["actual"]) ** 2).mean()))
mape = lambda c: float((abs(res[c] - res["actual"]) / res["actual"].replace(0, np.nan)).mean())
print(f"model    RMSE={rmse('pred'):.1f} MAPE={mape('pred'):.2%}")
print(f"baseline RMSE={rmse('baseline'):.1f} MAPE={mape('baseline'):.2%}")

fig, ax = plt.subplots(2, 1, figsize=(11, 8))
res[["actual", "pred", "baseline"]].plot(ax=ax[0]); ax[0].set_title(f"30-day revenue: actual vs model vs baseline ({a.country})")
ax[1].bar(["model", "baseline"], [rmse("pred"), rmse("baseline")]); ax[1].set_title("RMSE comparison")
plt.tight_layout(); plt.savefig(os.path.join(a.out, f"performance_{a.country}.png"))

# Relationship between model performance and the business metric (revenue level)
res["abs_pct_err"] = abs(res["pred"] - res["actual"]) / res["actual"].replace(0, np.nan)
res["rev_bucket"] = pd.qcut(res["actual"], 4, labels=["low", "mid-low", "mid-high", "high"])
print(res.groupby("rev_bucket", observed=True)["abs_pct_err"].mean())
fig, ax = plt.subplots(); ax.scatter(res["actual"], res["abs_pct_err"], s=8)
ax.set_xlabel("actual 30d revenue"); ax.set_ylabel("abs % error"); ax.set_title("Error vs business metric")
plt.savefig(os.path.join(a.out, f"error_vs_revenue_{a.country}.png"))
