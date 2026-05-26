import torch
import pytest
from nutrition5k_pkg.losses import multitask_mae_loss


def test_loss_direct_mode():
    tasks = ['calories', 'mass', 'fat', 'carb', 'protein']
    preds = {t: torch.tensor([10.0]) for t in tasks}
    targets = {t: torch.tensor([10.0]) for t in tasks}
    loss = multitask_mae_loss(preds, targets, tasks)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_loss_per_gram_mode():
    tasks = ['cal_per_g', 'fat_per_g', 'carb_per_g', 'protein_per_g']
    preds = {t: torch.tensor([1.0]) for t in tasks}
    targets = {t: torch.tensor([2.0]) for t in tasks}
    loss = multitask_mae_loss(preds, targets, tasks)
    # cal_per_g MAE = 1.0; mean(fat,carb,protein per_g MAE) = 1.0 → total = 2.0
    assert loss.item() == pytest.approx(2.0, abs=1e-4)


def test_loss_mass_only():
    preds = {'mass': torch.tensor([100.0])}
    targets = {'mass': torch.tensor([80.0])}
    loss = multitask_mae_loss(preds, targets, ['mass'])
    assert loss.item() == pytest.approx(20.0, abs=1e-4)
