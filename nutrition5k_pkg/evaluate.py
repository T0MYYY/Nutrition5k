import os, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from .metrics import compute_mae_report, print_results_table


def evaluate(model, config_path: str, output_dir: str):
    """Evaluate model on Nutri-Test. Saves per-dish predictions to CSV and prints MAE table."""
    from .train import load_config, _build_dataset

    cfg = load_config(config_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    test_ds = _build_dataset(cfg, 'test')
    nw = cfg['training'].get('num_workers', 0)
    pin = (device.type == 'cuda')
    loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'],
                        shuffle=False, num_workers=nw, pin_memory=pin)
    tasks = cfg['model']['tasks']

    all_preds   = {t: [] for t in tasks}
    all_targets = {t: [] for t in tasks}

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            if cfg['model']['type'] == 'mass_regressor':
                vol = labels.pop('volume', None)
                if vol is not None:
                    vol = vol.to(device)
                preds = {'mass': model(imgs, vol)}
            else:
                preds = model(imgs)

            for t in tasks:
                all_preds[t].extend(preds[t].cpu().numpy().tolist())
                all_targets[t].extend(labels[t].numpy().tolist())

    # Write predictions CSV
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, 'predictions.csv')
    with open(pred_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'pred_{t}' for t in tasks] + [f'gt_{t}' for t in tasks])
        for i in range(len(all_preds[tasks[0]])):
            row = [all_preds[t][i] for t in tasks] + [all_targets[t][i] for t in tasks]
            w.writerow(row)

    # Print MAE table
    results = {}
    for t in tasks:
        results[t] = compute_mae_report(
            np.array(all_preds[t]), np.array(all_targets[t])
        )
    print_results_table(results)
    return results
