import torch
import torch.nn.functional as F
from typing import Dict, List

_MACRO_KEYS = {'fat', 'carb', 'protein', 'fat_per_g', 'carb_per_g', 'protein_per_g'}
_CALORIE_KEYS = {'calories', 'cal_per_g'}

# Expected target magnitudes (paper Table 1; per_gram = total/mass = x/215)
# Dividing each MAE by its scale makes loss terms dimensionless so no single
# task dominates gradient (e.g. cal_per_g ~1.19 vs fat_per_g ~0.059 = 20x diff).
_TASK_SCALE = {
    'cal_per_g':     1.19,
    'fat_per_g':     0.059,
    'carb_per_g':    0.090,
    'protein_per_g': 0.084,
    'calories':    255.0,
    'fat':          12.7,
    'carb':         19.4,
    'protein':      18.0,
    'mass':        215.0,
}


def multitask_mae_loss(preds: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor],
                       task_names: List[str]) -> torch.Tensor:
    """Multi-task MAE loss per paper Eq. 3, with per-task scale normalisation.

    Each MAE term is divided by the expected target magnitude so that all tasks
    contribute comparably to the gradient regardless of unit differences.
    Calorie/calorie-per-gram: add normalised MAE directly.
    Macro nutrients: average normalised MAE across present macro tasks, then add.
    Mass: add normalised MAE directly.
    """
    total = torch.tensor(0.0, device=next(iter(preds.values())).device)
    macro_losses = []

    for name in task_names:
        scale = _TASK_SCALE.get(name, 1.0)
        mae = F.l1_loss(preds[name], targets[name]) / scale
        if name in _MACRO_KEYS:
            macro_losses.append(mae)
        else:
            total = total + mae

    if macro_losses:
        total = total + torch.stack(macro_losses).mean()

    return total


def geometric_l1_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    task_names: List[str],
    eps: float = 1e-6,
) -> torch.Tensor:
    losses = []
    for name in task_names:
        mae = F.l1_loss(preds[name], targets[name])
        losses.append(torch.clamp(mae, min=eps))
    product = torch.prod(torch.stack(losses))
    return product.pow(1.0 / len(task_names))
