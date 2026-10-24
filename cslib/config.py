"""Paths. Test runs are isolated from production models/logs via the `test` flag."""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "cs-train")
HORIZON = 30

def model_dir(test=False):
    d = os.path.join(BASE, "models", "test" if test else "prod")
    os.makedirs(d, exist_ok=True)
    return d

def log_dir(test=False):
    d = os.path.join(BASE, "logs", "test" if test else "prod")
    os.makedirs(d, exist_ok=True)
    return d
