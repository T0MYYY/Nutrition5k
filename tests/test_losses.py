import torch
import pytest
from nutrition5k_pkg.losses import geometric_l1_loss, multitask_mae_loss


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
    expected = (1.0 / 1.19) + ((1.0 / 0.059) + (1.0 / 0.090) + (1.0 / 0.084)) / 3.0
    assert loss.item() == pytest.approx(expected, abs=1e-4)


def test_loss_mass_only():
    preds = {'mass': torch.tensor([100.0])}
    targets = {'mass': torch.tensor([80.0])}
    loss = multitask_mae_loss(preds, targets, ['mass'])
    assert loss.item() == pytest.approx(20.0 / 215.0, abs=1e-4)


def test_geometric_l1_loss_matches_explicit_fifth_root():
    tasks = ['calories', 'mass', 'fat', 'carb', 'protein']
    preds = {
        'calories': torch.tensor([12.0, 16.0]),
        'mass': torch.tensor([8.0, 10.0]),
        'fat': torch.tensor([5.0, 9.0]),
        'carb': torch.tensor([22.0, 26.0]),
        'protein': torch.tensor([7.0, 13.0]),
    }
    targets = {
        'calories': torch.tensor([10.0, 10.0]),
        'mass': torch.tensor([4.0, 4.0]),
        'fat': torch.tensor([2.0, 2.0]),
        'carb': torch.tensor([20.0, 20.0]),
        'protein': torch.tensor([6.0, 6.0]),
    }

    expected = torch.prod(torch.stack([
        torch.nn.functional.l1_loss(preds[t], targets[t]) for t in tasks
    ])).pow(1.0 / 5.0)

    assert geometric_l1_loss(preds, targets, tasks).item() == pytest.approx(expected.item())


def test_geometric_l1_loss_stays_finite_when_a_task_matches_exactly():
    tasks = ['calories', 'mass', 'fat', 'carb', 'protein']
    preds = {t: torch.tensor([1.0], requires_grad=True) for t in tasks}
    targets = {t: torch.tensor([2.0]) for t in tasks}
    targets['mass'] = torch.tensor([1.0])

    loss = geometric_l1_loss(preds, targets, tasks)
    loss.backward()

    assert torch.isfinite(loss)
    for value in preds.values():
        assert torch.isfinite(value.grad).all()
