import os
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def aggregate_metrics(metrics_list):
    keys = metrics_list[0].keys()
    out = {}

    for key in keys:
        vals = [m[key] for m in metrics_list]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))

    return out