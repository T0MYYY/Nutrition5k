import numpy as np
from typing import Dict

# Dataset-level means from Table 1 of the paper
DATASET_MEANS = {
    'calories': 255.0,
    'mass':     215.0,
    'fat':       12.7,
    'carb':      19.4,
    'protein':   18.0,
}


def compute_mae_report(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Returns {'mae': float, 'mae_pct': float} where mae_pct = MAE / mean(targets) * 100."""
    mae = float(np.mean(np.abs(preds - targets)))
    mean_gt = float(np.mean(targets)) if np.mean(targets) != 0 else 1.0
    return {'mae': mae, 'mae_pct': mae / mean_gt * 100.0}


def print_results_table(results: Dict[str, Dict[str, float]]):
    """Pretty-print a Table-2-style results summary."""
    header = f"{'Metric':<12} {'MAE':>10} {'MAE %':>10}"
    print(header)
    print('-' * len(header))
    for metric, r in results.items():
        print(f"{metric:<12} {r['mae']:>10.2f} {r['mae_pct']:>9.1f}%")
