"""Data ingestion: reads ALL json files in a directory (so retraining is automatic)."""
import glob, os
import pandas as pd

RENAME = {"total_price": "price", "streamid": "stream_id", "timesviewed": "times_viewed",
          "invoice_id": "invoice"}

def load_json_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"no json files in {data_dir}")
    df = pd.concat([pd.read_json(f) for f in files], ignore_index=True)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=RENAME)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.drop_duplicates()
    return df

def top_countries(df, n=10):
    return df.groupby("country")["price"].sum().nlargest(n).index.tolist()

def convert_to_ts(df, country=None):
    """Daily time series (revenue, purchases, views) for a country or all countries."""
    d = df if country in (None, "all") else df[df["country"] == country]
    if d.empty:
        raise ValueError(f"unknown country: {country}")
    ts = d.groupby("date").agg(revenue=("price", "sum"), purchases=("invoice", "nunique"),
                               views=("times_viewed", "sum"))
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return ts.reindex(full, fill_value=0)

def fetch_ts(data_dir, country=None, cutoff=None):
    df = load_json_data(data_dir)
    if cutoff is not None:
        df = df[df["date"] <= pd.Timestamp(cutoff)]
    return convert_to_ts(df, country), df
