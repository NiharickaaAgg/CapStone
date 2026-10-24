"""EDA with visualizations. Usage: python scripts/eda.py data/cs-train"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cslib import ingest
d = ingest.load_json_data(sys.argv[1] if len(sys.argv) > 1 else "data/cs-train")
os.makedirs("reports", exist_ok=True)
d.groupby("country")["price"].sum().nlargest(10).plot.barh(title="Revenue by country (top 10)")
plt.tight_layout(); plt.savefig("reports/eda_country.png"); plt.clf()
ingest.convert_to_ts(d, "all")["revenue"].plot(title="Daily revenue, all countries")
plt.savefig("reports/eda_daily_revenue.png"); plt.clf()
ingest.convert_to_ts(d, "all")["revenue"].resample("M").sum().plot(title="Monthly revenue")
plt.savefig("reports/eda_monthly.png")
