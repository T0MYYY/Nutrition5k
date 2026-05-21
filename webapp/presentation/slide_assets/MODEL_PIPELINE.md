# Model pipeline (for slides)

Static overview of this repository’s calorie model. No execution required for this page.

## End-to-end flow

```mermaid
flowchart LR
  subgraph Data["Nutrition5k"]
    M[dish metadata → calories]
    I[overhead RGB / depth]
    S[official split IDs]
  end

  subgraph Prep["Dataset + transforms"]
    J[Train / val / test samples]
    R[Resize + ImageNet norm]
    Z[Optional 4ch RGB-D concat]
  end

  subgraph Model["CalorieRegressor"]
    B[ResNet-18 backbone]
    H[MLP head → scalar kcal]
    C[Optional Food-101 logits]
  end

  M --> J
  I --> J
  S --> J
  J --> R
  R --> Z
  Z --> B
  B --> H
  B --> C
```

## Training vs inference

- **Training**: SmoothL1 or MSE on targets (optional `log1p` target), ReduceLROnPlateau, early stopping on val MAE; optional auxiliary Food-101 cross-entropy on scheduled epochs.
- **Inference / evaluation**: Load `best.pt`, forward pass only; metrics from saved predictions require `evaluate.py` with `--save_predictions_csv` (still not retraining).

## Exporting this diagram

- GitHub renders Mermaid in `.md` files automatically.
- For Keynote/PowerPoint, paste the mermaid block into [mermaid.live](https://mermaid.live) and export PNG/SVG.
