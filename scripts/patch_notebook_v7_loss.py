#!/usr/bin/env python3
"""Patch notebook for v7 training: tanh-bounded outputs + recalibrated DirectionalLoss.

Changes:
  Cell 15 (model architectures): Add nn.Tanh() to each model's output head
  Cell 17 (DirectionalLoss): Recalibrate for tanh-bounded [-1,1] inputs

Run:  python3 scripts/patch_notebook_v7_loss.py
"""

import json
import sys
import shutil
from pathlib import Path

NOTEBOOK = Path('notebooks/retrain_98dim_colab.ipynb')
BACKUP = NOTEBOOK.with_suffix('.ipynb.bak_v6')


def patch_model_architectures(src: str) -> str:
    """Add nn.Tanh() to each model's final prediction layer."""

    # === TrainedQuantumTransformer ===
    # Before: nn.Linear(72, 1))
    # After:  nn.Linear(72, 1), nn.Tanh())
    src = src.replace(
        "nn.Linear(d_model, 144), nn.GELU(), nn.Dropout(0.2),\n"
        "            nn.Linear(144, 72), nn.GELU(), nn.Linear(72, 1))",
        "nn.Linear(d_model, 144), nn.GELU(), nn.Dropout(0.2),\n"
        "            nn.Linear(144, 72), nn.GELU(), nn.Linear(72, 1), nn.Tanh())"
    )

    # === TrainedBidirectionalLSTM ===
    # Before: nn.Linear(292, 146), nn.GELU(), nn.Linear(146, 1))
    # After:  nn.Linear(292, 146), nn.GELU(), nn.Linear(146, 1), nn.Tanh())
    src = src.replace(
        "nn.Linear(292, 146), nn.GELU(), nn.Linear(146, 1))",
        "nn.Linear(292, 146), nn.GELU(), nn.Linear(146, 1), nn.Tanh())"
    )

    # === TrainedDilatedCNN ===
    # classifier ends with nn.Linear(input_dim, 1))
    # Before: nn.Linear(input_dim, 1))
    # After:  nn.Linear(input_dim, 1), nn.Tanh())
    # Need to be specific since "input_dim, 1" appears in other contexts
    src = src.replace(
        "nn.Linear(166, input_dim), nn.BatchNorm1d(input_dim), nn.GELU(),\n"
        "            nn.Linear(input_dim, 1))",
        "nn.Linear(166, input_dim), nn.BatchNorm1d(input_dim), nn.GELU(),\n"
        "            nn.Linear(input_dim, 1), nn.Tanh())"
    )

    # === TrainedCNN ===
    # Before: nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
    # After:  nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Tanh())
    src = src.replace(
        "nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))",
        "nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Tanh())"
    )

    # === TrainedGRU ===
    # Before: nn.Linear(134, 64), nn.GELU(), nn.Linear(64, 1))
    # After:  nn.Linear(134, 64), nn.GELU(), nn.Linear(64, 1), nn.Tanh())
    src = src.replace(
        "nn.Linear(134, 64), nn.GELU(), nn.Linear(64, 1))",
        "nn.Linear(134, 64), nn.GELU(), nn.Linear(64, 1), nn.Tanh())"
    )

    # === TrainedMetaEnsemble ===
    # final_predictor ends with nn.Linear(32, 1))
    # Before: nn.Linear(64 + n_models, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))
    # After:  nn.Linear(64 + n_models, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1), nn.Tanh())
    src = src.replace(
        "nn.Linear(64 + n_models, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))",
        "nn.Linear(64 + n_models, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1), nn.Tanh())"
    )

    return src


def patch_directional_loss(src: str) -> str:
    """Replace DirectionalLoss v6 with v7: calibrated for tanh-bounded outputs."""

    new_loss = '''class DirectionalLoss(nn.Module):
    """v7: Recalibrated for tanh-bounded model outputs in [-1, 1].

    v6 was calibrated for unbounded micro-predictions (~0.01-0.05). With models
    now outputting tanh-bounded [-1, 1] values, the loss needs recalibration:

    - logit_scale: 20.0 -> 3.0 (tanh outputs are already in [-1,1], 3x gives
      logits in [-3,3] = ~5%-95% probability range. 20x gave [-20,20] = always
      99.99% confident, destroying gradient.)
    - margin: 0.10 -> 0.25 (with tanh outputs spanning [-1,1], require 0.25
      separation — meaningful directional commitment, not micro-signal)
    - mag_floor: 0.01 -> 0.10 (push |pred| above 0.10 — prevents collapse to
      zero while leaving room for low-confidence predictions)
    - mag_weight: 5.0 -> 3.0 (softer penalty since tanh naturally bounds outputs)

    At collapse (all pred=0), v7 penalty:
      BCE(0*3, 0.5) = 0.693 (uninformative)
      + 10 * relu(0.25 - 0) = 2.5 (strong anti-collapse)
      + 3.0 * relu(0.10 - 0) = 0.30
      Total = 3.49 (forces model away from zero quickly)
    """
    def __init__(self, logit_scale=3.0, margin=0.25):
        super().__init__()
        self.logit_scale = logit_scale
        self.margin = margin

    def forward(self, pred, target):
        pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        target = target.squeeze(-1) if target.dim() > 1 else target

        # 1. BCE direction: logit_scale=3.0 maps tanh output to reasonable probabilities
        #    pred=0.5 -> logit=1.5 -> 82% prob, pred=1.0 -> logit=3.0 -> 95% prob
        target_pos = (target > 0).float()
        bce = F.binary_cross_entropy_with_logits(
            pred * self.logit_scale, target_pos)

        # 2. Separation margin: require 0.25 gap between up/down predictions
        pos_mask = target > 0
        neg_mask = target <= 0
        if pos_mask.any() and neg_mask.any():
            separation = pred[pos_mask].mean() - pred[neg_mask].mean()
            sep_loss = F.relu(self.margin - separation)
        else:
            sep_loss = torch.tensor(0.0, device=pred.device)

        # 3. Magnitude floor: push |pred| above 0.10 (was 0.01)
        mag_loss = F.relu(0.10 - pred.abs()).mean()

        return bce + 10.0 * sep_loss + 3.0 * mag_loss'''

    # Find and replace the DirectionalLoss class (up to the next function def)
    lines = src.split('\n')
    new_lines = []
    in_class = False
    class_done = False

    for line in lines:
        if line.startswith('class DirectionalLoss'):
            in_class = True
            class_done = False
            continue
        if in_class and not class_done:
            # Look for the end of the class (next non-indented, non-empty line)
            if line and not line.startswith(' ') and not line.startswith('\t'):
                in_class = False
                class_done = True
                # Insert new class before this line
                new_lines.append(new_loss)
                new_lines.append('')
                new_lines.append(line)
            # Skip old class lines
            continue
        new_lines.append(line)

    result = '\n'.join(new_lines)

    # Also update the criterion instantiation comment in train_base_model reference
    # and the print at the end of the cell
    result = result.replace(
        "print('Training utilities defined (v6: BCE + 10*sep_margin(0.10) + 5*mag_floor)')",
        "print('Training utilities defined (v7: BCE*3 + 10*sep_margin(0.25) + 3*mag_floor(0.10))')"
    )

    return result


def patch_train_base_model(src: str) -> str:
    """Update criterion instantiation to use v7 parameters."""

    src = src.replace(
        "# v6 loss: BCE*20 + 10*separation_margin(0.10) + 5*magnitude_floor\n"
        "    criterion = DirectionalLoss(logit_scale=20.0, margin=0.10)",
        "# v7 loss: BCE*3 + 10*separation_margin(0.25) + 3*magnitude_floor(0.10)\n"
        "    # Recalibrated for tanh-bounded model outputs in [-1, 1]\n"
        "    criterion = DirectionalLoss(logit_scale=3.0, margin=0.25)"
    )

    return src


def main():
    if not NOTEBOOK.exists():
        print(f"ERROR: {NOTEBOOK} not found")
        sys.exit(1)

    # Backup
    if not BACKUP.exists():
        shutil.copy2(NOTEBOOK, BACKUP)
        print(f"Backup: {BACKUP}")

    nb = json.load(open(NOTEBOOK))

    def get_src(cell):
        s = cell['source']
        return ''.join(s) if isinstance(s, list) else s

    def set_src(cell, new_src):
        if isinstance(cell['source'], list):
            cell['source'] = new_src.split('\n')
            # nbformat expects each line (except last) to end with \n
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
        else:
            cell['source'] = new_src

    changes = 0

    # --- Cell 15: Model architectures ---
    src15 = get_src(nb['cells'][15])
    new15 = patch_model_architectures(src15)
    if new15 != src15:
        # Count tanh additions
        added = new15.count('nn.Tanh()') - src15.count('nn.Tanh()')
        print(f"Cell 15 (architectures): +{added} nn.Tanh() output layers")
        set_src(nb['cells'][15], new15)
        changes += 1
    else:
        print("Cell 15: NO CHANGES (patterns not matched)")

    # --- Cell 17: DirectionalLoss + training utils ---
    src17 = get_src(nb['cells'][17])
    new17 = patch_directional_loss(src17)
    if new17 != src17:
        print("Cell 17 (loss function): v6 -> v7 (logit_scale=3.0, margin=0.25, mag_floor=0.10)")
        set_src(nb['cells'][17], new17)
        changes += 1
    else:
        print("Cell 17: NO CHANGES (patterns not matched)")

    # --- Cell 23: train_base_model ---
    src23 = get_src(nb['cells'][23])
    new23 = patch_train_base_model(src23)
    if new23 != src23:
        print("Cell 23 (train_base_model): criterion updated to v7 params")
        set_src(nb['cells'][23], new23)
        changes += 1
    else:
        print("Cell 23: NO CHANGES (patterns not matched)")

    if changes > 0:
        json.dump(nb, open(NOTEBOOK, 'w'), indent=1, ensure_ascii=False)
        print(f"\nSaved {NOTEBOOK} ({changes} cells modified)")
    else:
        print("\nERROR: No changes applied — check pattern matching")
        sys.exit(1)


if __name__ == '__main__':
    main()
