import json, os, numpy as np, pandas as pd

def make_synthetic_data(out_dir, days=500, seed=0):
    """Writes JSON in the same shape as the case-study files (country, invoice, price, times_viewed, y/m/d)."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    rows, inv = [], 0
    for d in pd.date_range("2018-01-01", periods=days):
        for c in ["united_kingdom", "france", "germany"]:
            for _ in range(rng.integers(1, 6)):
                inv += 1
                rows.append({"country": c, "customer_id": int(rng.integers(1, 50)), "invoice": str(inv),
                             "price": float(rng.gamma(2, 10) * (1 + 0.3 * np.sin(d.dayofyear / 58))),
                             "stream_id": "s1", "times_viewed": int(rng.integers(1, 10)),
                             "year": d.year, "month": d.month, "day": d.day})
    half = len(rows) // 2
    json.dump(rows[:half], open(os.path.join(out_dir, "a.json"), "w"))
    json.dump(rows[half:], open(os.path.join(out_dir, "b.json"), "w"))
