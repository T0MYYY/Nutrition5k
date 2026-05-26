import numpy as np
import pytest
from nutrition5k_pkg.metrics import compute_mae_report


def test_mae_report_values():
    preds   = np.array([100., 200., 300.])
    targets = np.array([110., 190., 310.])
    result = compute_mae_report(preds, targets)
    assert result['mae'] == pytest.approx(10.0, abs=1e-4)
    # mean(targets) = 203.333...; 10/203.333*100 ≈ 4.918%
    assert result['mae_pct'] == pytest.approx(4.918032786885246, abs=1e-4)


def test_mae_report_perfect():
    arr = np.array([50., 100., 150.])
    result = compute_mae_report(arr, arr)
    assert result['mae'] == pytest.approx(0.0, abs=1e-6)
    assert result['mae_pct'] == pytest.approx(0.0, abs=1e-6)
