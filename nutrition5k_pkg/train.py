import os, time, csv
from typing import Dict, Any
import yaml
import torch
from torch.utils.data import DataLoader

from .data.metadata import load_dish_metadata, get_train_val_test_split, load_official_split
from .data.transforms import get_train_transform, get_val_transform, depth_to_tensor
from .data.dataset import Nutrition5kDataset
from .data.depth_dataset import DepthDataset
from .losses import multitask_mae_loss
from .models.multitask_head import MultitaskNutritionNet
from .models.mass_regressor import MassRegressor


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_volume_map(cfg: Dict) -> Dict[str, float]:
    """Load precomputed volume estimates from CSV, if configured."""
    vol_csv = cfg['data'].get('volume_csv')
    if not vol_csv or not os.path.isfile(vol_csv):
        return {}
    with open(vol_csv) as f:
        return {row['dish_id']: float(row['volume_cm3']) for row in csv.DictReader(f)}


def _build_dataset(cfg: Dict, split: str):
    meta = load_dish_metadata(
        cfg['data']['cafe1_csv'],
        cfg['data'].get('cafe2_csv'),
    )
    if cfg['data'].get('train_ids_txt'):
        train_ids, val_ids, test_ids = load_official_split(
            cfg['data']['train_ids_txt'],
            cfg['data']['test_ids_txt'],
            val_ratio=cfg['data'].get('val_ratio', 0.1),
        )
    else:
        train_ids, val_ids, test_ids = get_train_val_test_split(
            list(meta.keys()),
            val_ratio=cfg['data'].get('val_ratio', 0.1),
            test_ratio=cfg['data'].get('test_ratio', 0.1),
            seed=cfg['data'].get('split_seed', 42),
        )
    ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}[split]
    # Filter to dishes that have metadata (official split may include cafe2 dishes)
    ids = [d for d in ids if d in meta]
    mode = cfg['model']['mode']
    exp_type = cfg['model']['type']
    is_train = (split == 'train')
    pre_resized = cfg['data'].get('pre_resized', False)
    rgb_tfm = get_train_transform(pre_resized) if is_train else get_val_transform(pre_resized)

    preload = cfg['data'].get('preload_ram', False) and is_train

    if exp_type == 'mass_regressor':
        return DepthDataset(
            overhead_root=cfg['data']['overhead_root'],
            metadata=meta, dish_ids=ids, mode=mode,
            rgb_transform=rgb_tfm,
            depth_transform=None,
            volume_map=_load_volume_map(cfg),
            preload_ram=preload,
        )
    elif cfg['data'].get('use_depth', False):
        return DepthDataset(
            overhead_root=cfg['data']['overhead_root'],
            metadata=meta, dish_ids=ids, mode=mode,
            rgb_transform=rgb_tfm,
            depth_transform=depth_to_tensor(256),
            preload_ram=preload,
        )
    else:
        return Nutrition5kDataset(
            side_angle_root=cfg['data']['side_angle_root'],
            metadata=meta, dish_ids=ids, mode=mode,
            transform=rgb_tfm,
            preload_ram=preload,
        )


def _build_model(cfg):
    exp_type = cfg['model']['type']
    tasks = cfg['model']['tasks']
    backbone_name = cfg['model'].get('backbone', 'inception_v3')
    if exp_type == 'multitask':
        in_channels = 4 if cfg['data'].get('use_depth', False) else 3
        return MultitaskNutritionNet(tasks=tasks, in_channels=in_channels,
                                     backbone_name=backbone_name)
    if exp_type == 'mass_regressor':
        return MassRegressor(use_volume=cfg['model'].get('use_volume', True),
                             backbone_name=backbone_name)
    raise ValueError(f"Unknown model type: {exp_type}")


def run(config_path: str, resume_from: str = None):
    cfg = load_config(config_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_ds = _build_dataset(cfg, 'train')
    val_ds   = _build_dataset(cfg, 'val')
    bs = cfg['training']['batch_size']
    nw = cfg['training'].get('num_workers', 10)
    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin,
                              prefetch_factor=2 if nw > 0 else None,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              prefetch_factor=2 if nw > 0 else None,
                              persistent_workers=(nw > 0))

    model = _build_model(cfg).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    if cfg['training'].get('use_compile', False) and torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    if resume_from:
        state = torch.load(resume_from, map_location=device)
        model.load_state_dict(state)
        print(f"Resumed weights from {resume_from}")

    # Skip freeze phase when resuming — backbone is already pre-trained
    freeze_epochs = 0 if resume_from else cfg['training'].get('freeze_backbone_epochs', 0)
    head_lr = cfg['training'].get('head_lr', cfg['training']['lr'])
    rmsprop_kwargs = dict(
        momentum=cfg['training']['momentum'],
        alpha=cfg['training']['alpha'],
        eps=cfg['training']['eps'],
    )

    opt_type = cfg['training'].get('optimizer', 'adam').lower()

    def _make_optimizer(params, lr):
        if opt_type == 'adam':
            return torch.optim.Adam(params, lr=lr,
                                    weight_decay=cfg['training'].get('weight_decay', 1e-4))
        return torch.optim.RMSprop(params, lr=lr, **rmsprop_kwargs)

    def _make_scheduler(opt):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min',
            factor=cfg['training'].get('lr_factor', 0.5),
            patience=cfg['training'].get('lr_patience', 5),
            min_lr=cfg['training'].get('lr_min', 1e-6),
        )

    if freeze_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = _make_optimizer(head_params, head_lr)
        print(f"Phase 1: backbone frozen, training head only at lr={head_lr:.0e} for {freeze_epochs} epochs")
    else:
        optimizer = _make_optimizer(model.parameters(), cfg['training']['lr'])
    scheduler = _make_scheduler(optimizer)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    tasks = cfg['model']['tasks']
    ckpt_dir = cfg['training']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    patience = cfg['training'].get('early_stopping_patience', None)
    max_epochs = cfg['training'].get('max_epochs', 10_000)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    log_path = os.path.join(ckpt_dir, 'train_log.csv')

    with open(log_path, 'w', newline='') as logf:
        writer = csv.writer(logf)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'elapsed_s'])

        for epoch in range(1, max_epochs + 1):
            # Phase 2: unfreeze backbone after freeze_epochs
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                for p in model.backbone.parameters():
                    p.requires_grad = True
                optimizer = _make_optimizer(model.parameters(), cfg['training']['lr'])
                scheduler = _make_scheduler(optimizer)
                best_val_loss = float('inf')
                epochs_no_improve = 0
                print(f"Phase 2: backbone unfrozen, full fine-tuning at lr={cfg['training']['lr']:.0e}")

            t0 = time.time()
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                imgs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                if device.type == 'cuda':
                    imgs = imgs.to(memory_format=torch.channels_last)
                labels_d = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    if cfg['model']['type'] == 'mass_regressor':
                        vol = labels_d.pop('volume', None)
                        preds = {'mass': model(imgs, vol)}
                        labels_d = {'mass': labels_d['mass']}
                    else:
                        preds = model(imgs)
                    loss = multitask_mae_loss(preds, labels_d, tasks)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    imgs, labels = batch
                    imgs = imgs.to(device, non_blocking=True)
                    if device.type == 'cuda':
                        imgs = imgs.to(memory_format=torch.channels_last)
                    labels_d = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        if cfg['model']['type'] == 'mass_regressor':
                            vol = labels_d.pop('volume', None)
                            preds = {'mass': model(imgs, vol)}
                            labels_d = {'mass': labels_d['mass']}
                        else:
                            preds = model(imgs)
                        val_loss += multitask_mae_loss(preds, labels_d, tasks).item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            elapsed = time.time() - t0
            writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}', f'{current_lr:.2e}', f'{elapsed:.1f}'])
            logf.flush()
            print(f"Epoch {epoch:4d} | train {train_loss:.3f} | val {val_loss:.3f} | lr {current_lr:.1e} | {elapsed:.0f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pt'))
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return os.path.join(ckpt_dir, 'best.pt')
