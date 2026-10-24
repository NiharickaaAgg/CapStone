"""CSV logging for train and predict. test=True writes to an isolated test directory."""
import csv, os, time
from datetime import date
from cslib.config import log_dir

TRAIN_FIELDS = ["timestamp", "country", "best_model", "rmse", "baseline_rmse", "runtime", "version", "note"]
PRED_FIELDS = ["timestamp", "country", "target_date", "y_pred", "runtime", "version"]

def _write(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        w.writerow(row)

def update_train_log(country, best_model, rmse, baseline_rmse, runtime, version, note="", test=False):
    path = os.path.join(log_dir(test), f"train-{date.today().strftime('%Y-%m')}.log")
    _write(path, TRAIN_FIELDS, [time.time(), country, best_model, rmse, baseline_rmse, runtime, version, note])

def update_predict_log(country, target_date, y_pred, runtime, version, test=False):
    path = os.path.join(log_dir(test), f"predict-{date.today().strftime('%Y-%m')}.log")
    _write(path, PRED_FIELDS, [time.time(), country, target_date, y_pred, runtime, version])
