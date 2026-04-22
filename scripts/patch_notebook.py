"""
Patch retrain_98dim_colab.ipynb for Spec 16:
  Part A: Add LightGBM training cell (Section 6b), update constants & meta-ensemble
  Part C: Fix derivatives data fetching (windowed pagination, DERIV_DAYS=180)
  Part D: Add confidence-stratified accuracy analysis cell (Section 9b)
  Part E: Update zip/download cell to include .pkl and .json files
"""

import json
import sys
import os

NB_PATH = 'notebooks/retrain_98dim_colab.ipynb'


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def source_to_str(cell):
    return ''.join(cell['source'])


def str_to_source(s):
    lines = s.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        elif line:
            result.append(line)
    return result


def patch_cell_10_constants(cell):
    src = source_to_str(cell)
    src = src.replace(
        "N_BASE_MODELS = 5",
        "N_BASE_MODELS = 6"
    )
    src = src.replace(
        "BASE_MODEL_NAMES = ['quantum_transformer', 'bidirectional_lstm', 'dilated_cnn', 'cnn', 'gru']",
        "BASE_MODEL_NAMES = ['quantum_transformer', 'bidirectional_lstm', 'dilated_cnn', 'cnn', 'gru', 'lightgbm']"
    )
    cell['source'] = str_to_source(src)
    return cell


def patch_cell_15_meta_ensemble(cell):
    src = source_to_str(cell)
    src = src.replace(
        "    def __init__(self, input_dim=INPUT_DIM):\n"
        "        super().__init__()\n"
        "        self._input_dim = input_dim\n"
        "        n_models = N_BASE_MODELS",
        "    def __init__(self, input_dim=INPUT_DIM, n_models=N_BASE_MODELS):\n"
        "        super().__init__()\n"
        "        self._input_dim = input_dim\n"
        "        self._n_models = n_models"
    )
    src = src.replace(
        "meta_test = torch.randn(2, INPUT_DIM + N_BASE_MODELS)\n"
        "m = TrainedMetaEnsemble(input_dim=INPUT_DIM)\n"
        "print(f'MetaEnsemble: output {m(meta_test)[0].shape}')",
        "meta_test = torch.randn(2, INPUT_DIM + N_BASE_MODELS)\n"
        "m = TrainedMetaEnsemble(input_dim=INPUT_DIM, n_models=N_BASE_MODELS)\n"
        "print(f'MetaEnsemble: output {m(meta_test)[0].shape}')"
    )
    cell['source'] = str_to_source(src)
    return cell


def patch_cell_24_base_configs(cell):
    src = source_to_str(cell)
    src = src.replace(
        "print(f'Base models trained: {len(trained_base_models)}/5')",
        "print(f'Base models trained: {len(trained_base_models)}/{len(base_configs)}')"
    )
    cell['source'] = str_to_source(src)
    return cell


def create_lightgbm_cell():
    lines = [
        "# ================================================================\n",
        "# Section 6b: Train LightGBM (gradient-boosted trees)\n",
        "# ================================================================\n",
        "# LightGBM is structurally different from neural nets -- trees find\n",
        "# interaction effects and threshold rules that DL models miss.\n",
        "# This makes it an excellent diversifier in the meta-ensemble.\n",
        "# ================================================================\n",
        "\n",
        "import lightgbm as lgb\n",
        "import pickle\n",
        "\n",
        "print(f'\\n{\"=\"*60}')\n",
        "print('Training LightGBM (gradient-boosted trees)')\n",
        "print(f'{\"=\"*60}')\n",
        "\n",
        "def _prepare_lgb_features(X_seq):\n",
        "    # Flatten sequence data for LightGBM: [last, mean, std] -> (N, INPUT_DIM*3)\n",
        "    last = X_seq[:, -1, :]           # (N, INPUT_DIM)\n",
        "    mean = X_seq.mean(axis=1)        # (N, INPUT_DIM)\n",
        "    std  = X_seq.std(axis=1)         # (N, INPUT_DIM)\n",
        "    return np.concatenate([last, mean, std], axis=1)  # (N, INPUT_DIM*3)\n",
        "\n",
        "lgb_X_train = _prepare_lgb_features(X_train)\n",
        "lgb_X_val   = _prepare_lgb_features(X_val)\n",
        "lgb_X_test  = _prepare_lgb_features(X_test)\n",
        "\n",
        "# Binary classification: is forward return positive?\n",
        "lgb_y_train = (y_train > 0).astype(int)\n",
        "lgb_y_val   = (y_val > 0).astype(int)\n",
        "lgb_y_test  = (y_test > 0).astype(int)\n",
        "\n",
        "print(f'  LGB features: {lgb_X_train.shape} (last + mean + std of {INPUT_DIM}-dim sequence)')\n",
        "print(f'  Label balance: train={lgb_y_train.mean():.3f}, val={lgb_y_val.mean():.3f}, test={lgb_y_test.mean():.3f}')\n",
        "\n",
        "lgb_train = lgb.Dataset(lgb_X_train, label=lgb_y_train)\n",
        "lgb_val   = lgb.Dataset(lgb_X_val, label=lgb_y_val, reference=lgb_train)\n",
        "\n",
        "lgb_params = {\n",
        "    'objective': 'binary',\n",
        "    'metric': 'binary_logloss',\n",
        "    'boosting_type': 'gbdt',\n",
        "    'learning_rate': 0.03,\n",
        "    'num_leaves': 63,\n",
        "    'max_depth': 7,\n",
        "    'min_child_samples': 50,\n",
        "    'subsample': 0.8,\n",
        "    'colsample_bytree': 0.6,\n",
        "    'reg_alpha': 0.1,\n",
        "    'reg_lambda': 1.0,\n",
        "    'verbose': -1,\n",
        "    'seed': 42,\n",
        "}\n",
        "\n",
        "callbacks = [\n",
        "    lgb.early_stopping(stopping_rounds=30),\n",
        "    lgb.log_evaluation(period=50),\n",
        "]\n",
        "\n",
        "t0 = time.time()\n",
        "lgb_model = lgb.train(\n",
        "    lgb_params,\n",
        "    lgb_train,\n",
        "    valid_sets=[lgb_val],\n",
        "    valid_names=['val'],\n",
        "    num_boost_round=500,\n",
        "    callbacks=callbacks,\n",
        ")\n",
        "lgb_elapsed = time.time() - t0\n",
        "\n",
        "# Evaluate\n",
        "lgb_val_prob = lgb_model.predict(lgb_X_val)\n",
        "lgb_test_prob = lgb_model.predict(lgb_X_test)\n",
        "\n",
        "# Convert probabilities to signed predictions: (prob - 0.5) * 2 -> [-1, 1]\n",
        "lgb_val_pred  = np.clip((lgb_val_prob  - 0.5) * 2.0, -1.0, 1.0)\n",
        "lgb_test_pred = np.clip((lgb_test_prob - 0.5) * 2.0, -1.0, 1.0)\n",
        "\n",
        "lgb_val_acc  = directional_accuracy(lgb_val_pred, y_val)\n",
        "lgb_test_acc = directional_accuracy(lgb_test_pred, y_test)\n",
        "lgb_pred_std = float(np.std(lgb_test_pred))\n",
        "\n",
        "print(f'\\n  LightGBM results:')\n",
        "print(f'    Val dir_acc:  {lgb_val_acc:.3f}')\n",
        "print(f'    Test dir_acc: {lgb_test_acc:.3f}')\n",
        "print(f'    Pred std:     {lgb_pred_std:.4f}')\n",
        "print(f'    Best round:   {lgb_model.best_iteration}')\n",
        "print(f'    Time:         {lgb_elapsed:.1f}s')\n",
        "\n",
        "# Feature importance (top 20)\n",
        "importance = lgb_model.feature_importance(importance_type='gain')\n",
        "feat_names = [f'feat_{i}' for i in range(lgb_X_train.shape[1])]\n",
        "# Label the three sections\n",
        "for i in range(INPUT_DIM):\n",
        "    feat_names[i] = f'last_{i}'\n",
        "    feat_names[INPUT_DIM + i] = f'mean_{i}'\n",
        "    feat_names[INPUT_DIM * 2 + i] = f'std_{i}'\n",
        "top_idx = np.argsort(importance)[::-1][:20]\n",
        "print(f'\\n  Top 20 features by gain:')\n",
        "for rank, idx in enumerate(top_idx):\n",
        "    print(f'    {rank+1:2d}. {feat_names[idx]:>12s}: {importance[idx]:.0f}')\n",
        "\n",
        "# Save LightGBM model\n",
        "lgb_pkl_path = 'models/trained/best_lightgbm_model.pkl'\n",
        "lgb_meta_path = 'models/trained/lightgbm_meta.json'\n",
        "\n",
        "with open(lgb_pkl_path, 'wb') as f:\n",
        "    pickle.dump(lgb_model, f)\n",
        "print(f'\\n  Saved: {lgb_pkl_path} ({os.path.getsize(lgb_pkl_path)/1e6:.1f} MB)')\n",
        "\n",
        "# Save metadata for local bot\n",
        "lgb_meta = {\n",
        "    'input_dim': INPUT_DIM,\n",
        "    'n_features': lgb_X_train.shape[1],\n",
        "    'feature_prep': 'last_mean_std',\n",
        "    'objective': 'binary',\n",
        "    'best_iteration': lgb_model.best_iteration,\n",
        "    'val_acc': float(lgb_val_acc),\n",
        "    'test_acc': float(lgb_test_acc),\n",
        "}\n",
        "with open(lgb_meta_path, 'w') as f:\n",
        "    json.dump(lgb_meta, f, indent=2)\n",
        "print(f'  Saved: {lgb_meta_path}')\n",
        "\n",
        "# Register in trained_base_models for meta-ensemble\n",
        "trained_base_models['lightgbm'] = lgb_model\n",
        "\n",
        "results['lightgbm'] = {\n",
        "    'val_loss': float(lgb_model.best_score['val']['binary_logloss']),\n",
        "    'val_acc': lgb_val_acc,\n",
        "    'test_acc': lgb_test_acc,\n",
        "    'test_pred_std': lgb_pred_std,\n",
        "    'epochs': lgb_model.best_iteration,\n",
        "    'time_min': lgb_elapsed / 60,\n",
        "}\n",
        "\n",
        "print(f'\\n  LightGBM registered as base model #{len(trained_base_models)}')\n",
        "print(f'  trained_base_models keys: {list(trained_base_models.keys())}')",
    ]
    return make_code_cell(lines)


def patch_cell_26_meta_inputs(cell):
    src = source_to_str(cell)

    # Fix the assertion
    src = src.replace(
        "assert len(trained_base_models) == 5, f'Need all 5 base models, got {len(trained_base_models)}'",
        "assert len(trained_base_models) == N_BASE_MODELS, f'Need all {N_BASE_MODELS} base models, got {len(trained_base_models)}: {list(trained_base_models.keys())}'"
    )

    # Add LightGBM handling inside the loop
    old_loop = (
        "    for name in BASE_MODEL_NAMES:\n"
        "        model = base_models[name]\n"
        "        model_preds = []"
    )
    new_loop = (
        "    for name in BASE_MODEL_NAMES:\n"
        "        model = base_models[name]\n"
        "        if name == 'lightgbm':\n"
        "            # LightGBM uses flattened features, not sequence tensors\n"
        "            lgb_feats = _prepare_lgb_features(X)\n"
        "            probs = model.predict(lgb_feats)\n"
        "            all_preds[name] = np.clip((probs - 0.5) * 2.0, -1.0, 1.0)\n"
        "            acc = directional_accuracy(all_preds[name], y)\n"
        "            print(f'  {name}: dir_acc={acc:.3f}')\n"
        "            continue\n"
        "        model_preds = []"
    )

    if old_loop in src:
        src = src.replace(old_loop, new_loop)
    else:
        print("WARNING: Could not find exact loop match in Cell 26")

    # Update the meta_X shape print
    src = src.replace(
        "    print(f'Meta-inputs: {meta_X.shape}')",
        "    print(f'Meta-inputs: {meta_X.shape} (features={INPUT_DIM} + {N_BASE_MODELS} model preds)')"
    )

    cell['source'] = str_to_source(src)
    return cell


def patch_cell_7_derivatives(cell):
    src = source_to_str(cell)

    # Fix DERIV_DAYS
    src = src.replace(
        "DERIV_DAYS = 730  # 2 years of history",
        "DERIV_DAYS = 180  # ~6 months -- Binance OI/LS endpoints limited to ~30d windows"
    )

    # Replace the _paginate function with windowed version
    old_paginate = (
        "def _paginate(url, symbol, period, start_ms, end_ms, val_key, label):\n"
        "    rows = []\n"
        "    cur = start_ms\n"
        "    while cur < end_ms:\n"
        "        resp = requests.get(url, params={\n"
        "            'symbol': symbol, 'period': period,\n"
        "            'startTime': cur, 'endTime': end_ms, 'limit': 500,\n"
        "        }, timeout=15)\n"
        "        if resp.status_code != 200:\n"
        "            print(f'    {label} HTTP {resp.status_code}: {resp.text[:100]}')\n"
        "            break\n"
        "        data = resp.json()\n"
        "        if not data:\n"
        "            break\n"
        "        for e in data:\n"
        "            rows.append({'timestamp': int(e['timestamp'])//1000, 'value': float(e[val_key])})\n"
        "        cur = int(data[-1]['timestamp']) + 1\n"
        "        _time.sleep(REQ_DELAY)\n"
        "    return pd.DataFrame(rows).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True) if rows else pd.DataFrame()"
    )

    new_paginate = (
        "WINDOW_MS = 30 * 86400 * 1000  # 30-day windows for OI/LS endpoints\n"
        "\n"
        "def _paginate(url, symbol, period, start_ms, end_ms, val_key, label):\n"
        "    # Windowed pagination -- Binance OI/LS endpoints return max ~30 days per request\n"
        "    rows = []\n"
        "    window_start = start_ms\n"
        "    while window_start < end_ms:\n"
        "        window_end = min(window_start + WINDOW_MS, end_ms)\n"
        "        cur = window_start\n"
        "        while cur < window_end:\n"
        "            resp = requests.get(url, params={\n"
        "                'symbol': symbol, 'period': period,\n"
        "                'startTime': cur, 'endTime': window_end, 'limit': 500,\n"
        "            }, timeout=15)\n"
        "            if resp.status_code != 200:\n"
        "                print(f'    {label} HTTP {resp.status_code} (window {window_start}-{window_end}): {resp.text[:100]}')\n"
        "                break\n"
        "            data = resp.json()\n"
        "            if not data:\n"
        "                break\n"
        "            for e in data:\n"
        "                rows.append({'timestamp': int(e['timestamp'])//1000, 'value': float(e[val_key])})\n"
        "            cur = int(data[-1]['timestamp']) + 1\n"
        "            _time.sleep(REQ_DELAY)\n"
        "        window_start = window_end\n"
        "    return pd.DataFrame(rows).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True) if rows else pd.DataFrame()"
    )

    if old_paginate in src:
        src = src.replace(old_paginate, new_paginate)
    else:
        print("WARNING: Could not find exact _paginate match in Cell 7")

    cell['source'] = str_to_source(src)
    return cell


def create_confidence_analysis_cell():
    lines = [
        "# ================================================================\n",
        "# Section 9b: Confidence-Stratified Accuracy Analysis\n",
        "# ================================================================\n",
        "# Does model confidence actually correlate with accuracy?\n",
        "# If high-confidence predictions aren't more accurate, the bot's\n",
        "# confidence filtering is useless. This analysis validates the\n",
        "# signal before deploying.\n",
        "# ================================================================\n",
        "\n",
        "print(f'\\n{\"=\"*60}')\n",
        "print('CONFIDENCE-STRATIFIED ACCURACY ANALYSIS')\n",
        "print(f'{\"=\"*60}')\n",
        "\n",
        "def confidence_stratified_analysis(model_name, predictions, targets, n_quintiles=5):\n",
        "    predictions = np.array(predictions).flatten()\n",
        "    targets = np.array(targets).flatten()\n",
        "    confidence = np.abs(predictions)\n",
        "    correct = ((predictions > 0) & (targets > 0)) | ((predictions < 0) & (targets < 0))\n",
        "    try:\n",
        "        quintile_edges = np.percentile(confidence, np.linspace(0, 100, n_quintiles + 1))\n",
        "        quintile_edges = np.unique(quintile_edges)\n",
        "        if len(quintile_edges) < 3:\n",
        "            print(f'  {model_name}: insufficient confidence spread for quintile analysis')\n",
        "            return None\n",
        "    except Exception as e:\n",
        "        print(f'  {model_name}: quintile computation failed: {e}')\n",
        "        return None\n",
        "    rows = []\n",
        "    for i in range(len(quintile_edges) - 1):\n",
        "        lo, hi = quintile_edges[i], quintile_edges[i + 1]\n",
        "        if i == len(quintile_edges) - 2:\n",
        "            mask = (confidence >= lo) & (confidence <= hi)\n",
        "        else:\n",
        "            mask = (confidence >= lo) & (confidence < hi)\n",
        "        n = mask.sum()\n",
        "        if n < 10:\n",
        "            continue\n",
        "        acc = correct[mask].mean()\n",
        "        avg_conf = confidence[mask].mean()\n",
        "        rows.append({\n",
        "            'quintile': i + 1,\n",
        "            'conf_range': f'{lo:.3f}-{hi:.3f}',\n",
        "            'n_samples': int(n),\n",
        "            'accuracy': float(acc),\n",
        "            'avg_confidence': float(avg_conf),\n",
        "        })\n",
        "    return rows\n",
        "\n",
        "# Collect predictions from all models on test set\n",
        "print('\\nCollecting test-set predictions from all models...\\n')\n",
        "\n",
        "model_test_preds = {}\n",
        "\n",
        "# DL models\n",
        "for name in ['quantum_transformer', 'bidirectional_lstm', 'dilated_cnn', 'cnn', 'gru']:\n",
        "    if name not in trained_base_models:\n",
        "        continue\n",
        "    model = trained_base_models[name]\n",
        "    model.eval()\n",
        "    preds_list = []\n",
        "    for i in range(0, len(X_test), 128):\n",
        "        batch = torch.FloatTensor(X_test[i:i+128]).to(device)\n",
        "        with torch.no_grad():\n",
        "            output = model(batch)\n",
        "            pred = output[0] if isinstance(output, tuple) else output\n",
        "            preds_list.append(torch.tanh(pred.squeeze(-1)).cpu().numpy())\n",
        "    model_test_preds[name] = np.concatenate(preds_list)\n",
        "\n",
        "# LightGBM\n",
        "if 'lightgbm' in trained_base_models:\n",
        "    lgb_feats_test = _prepare_lgb_features(X_test)\n",
        "    lgb_probs = trained_base_models['lightgbm'].predict(lgb_feats_test)\n",
        "    model_test_preds['lightgbm'] = np.clip((lgb_probs - 0.5) * 2.0, -1.0, 1.0)\n",
        "\n",
        "# Meta-ensemble\n",
        "try:\n",
        "    meta_preds_list = []\n",
        "    meta_test_loader_q = DataLoader(\n",
        "        TensorDataset(torch.FloatTensor(meta_X_test), torch.FloatTensor(meta_y_test)),\n",
        "        batch_size=128, shuffle=False)\n",
        "    meta_model.eval()\n",
        "    with torch.no_grad():\n",
        "        for X_b, y_b in meta_test_loader_q:\n",
        "            pred, conf = meta_model(X_b.to(device))\n",
        "            meta_preds_list.append(torch.tanh(pred.squeeze(-1)).cpu().numpy())\n",
        "    model_test_preds['meta_ensemble'] = np.concatenate(meta_preds_list)\n",
        "except Exception as e:\n",
        "    print(f'  Meta-ensemble predictions failed: {e}')\n",
        "\n",
        "# Run analysis for each model\n",
        "print(f'{\"Model\":<22s} {\"Quintile\":>8s} {\"Conf Range\":>14s} {\"N\":>6s} {\"Accuracy\":>10s} {\"AvgConf\":>8s}')\n",
        "print('-' * 72)\n",
        "\n",
        "all_strat_results = {}\n",
        "for name, preds in model_test_preds.items():\n",
        "    rows = confidence_stratified_analysis(name, preds, y_test)\n",
        "    if rows is None:\n",
        "        continue\n",
        "    all_strat_results[name] = rows\n",
        "    for r in rows:\n",
        "        print(f'{name:<22s} {r[\"quintile\"]:>8d} {r[\"conf_range\"]:>14s} {r[\"n_samples\"]:>6d} '\n",
        "              f'{r[\"accuracy\"]:>10.3f} {r[\"avg_confidence\"]:>8.3f}')\n",
        "    # Check monotonicity\n",
        "    accs = [r['accuracy'] for r in rows]\n",
        "    if len(accs) >= 3:\n",
        "        increasing = sum(1 for i in range(1, len(accs)) if accs[i] > accs[i-1])\n",
        "        total = len(accs) - 1\n",
        "        mono_score = increasing / total if total > 0 else 0\n",
        "        verdict = 'GOOD' if mono_score >= 0.6 else 'WEAK' if mono_score >= 0.3 else 'BAD'\n",
        "        print(f'  -> Monotonicity: {increasing}/{total} ({mono_score:.0%}) -- {verdict}')\n",
        "    print()\n",
        "\n",
        "# Overall summary\n",
        "print(f'\\n{\"=\"*60}')\n",
        "print('CONFIDENCE FILTERING VERDICT')\n",
        "print(f'{\"=\"*60}')\n",
        "good_models, weak_models, bad_models = [], [], []\n",
        "for name, rows in all_strat_results.items():\n",
        "    accs = [r['accuracy'] for r in rows]\n",
        "    if len(accs) < 3:\n",
        "        continue\n",
        "    spread = accs[-1] - accs[0]\n",
        "    if spread > 0.03:\n",
        "        good_models.append((name, spread))\n",
        "    elif spread > 0.01:\n",
        "        weak_models.append((name, spread))\n",
        "    else:\n",
        "        bad_models.append((name, spread))\n",
        "\n",
        "if good_models:\n",
        "    print(f'\\nModels where confidence filtering HELPS (top-bottom spread > 3pp):')\n",
        "    for name, spread in good_models:\n",
        "        print(f'  {name}: {spread*100:+.1f}pp')\n",
        "if weak_models:\n",
        "    print(f'\\nModels with WEAK confidence signal (1-3pp spread):')\n",
        "    for name, spread in weak_models:\n",
        "        print(f'  {name}: {spread*100:+.1f}pp')\n",
        "if bad_models:\n",
        "    print(f'\\nModels where confidence filtering is USELESS (<1pp spread):')\n",
        "    for name, spread in bad_models:\n",
        "        print(f'  {name}: {spread*100:+.1f}pp')\n",
        "\n",
        "print(f'\\nRecommendation: Set confidence threshold based on the meta-ensemble\\'s')\n",
        "print(f'top quintile accuracy. If meta-ensemble shows good monotonicity,')\n",
        "print(f'filter predictions below the 40th percentile confidence.')",
    ]
    return make_code_cell(lines)


def patch_cell_31_summary(cell):
    src = source_to_str(cell)
    src = src.replace(
        "for name in [n for n, _ in base_configs] + ['meta_ensemble', 'vae']:",
        "for name in [n for n, _ in base_configs] + ['lightgbm', 'meta_ensemble', 'vae']:"
    )
    cell['source'] = str_to_source(src)
    return cell


def patch_cell_32_zip(cell):
    src = source_to_str(cell)
    src = src.replace(
        "if f.endswith('.pth'):",
        "if f.endswith(('.pth', '.pkl', '.json')):"
    )
    cell['source'] = str_to_source(src)
    return cell


def main():
    with open(NB_PATH) as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"Original notebook: {len(cells)} cells")

    # ---- Part A: Update constants (Cell 10) ----
    print("\n[Part A] Patching Cell 10: N_BASE_MODELS=6, BASE_MODEL_NAMES += 'lightgbm'")
    cells[10] = patch_cell_10_constants(cells[10])

    # ---- Part A: Update TrainedMetaEnsemble (Cell 15) ----
    print("[Part A] Patching Cell 15: TrainedMetaEnsemble(n_models=N_BASE_MODELS)")
    cells[15] = patch_cell_15_meta_ensemble(cells[15])

    # ---- Part A: Update base_configs cell (Cell 24) ----
    print("[Part A] Patching Cell 24: dynamic base model count in print")
    cells[24] = patch_cell_24_base_configs(cells[24])

    # ---- Part A: Insert LightGBM training cell after Cell 24 ----
    print("[Part A] Inserting new cell after Cell 24: Section 6b LightGBM training")
    lgb_cell = create_lightgbm_cell()
    cells.insert(25, lgb_cell)
    # After insertion: old Cell 25 -> Cell 26, old Cell 26 -> Cell 27, etc.

    # ---- Part A: Update meta-ensemble assertion (now Cell 27, was Cell 26) ----
    print("[Part A] Patching Cell 27 (was 26): meta-ensemble assertion + LightGBM")
    cells[27] = patch_cell_26_meta_inputs(cells[27])

    # ---- Part C: Fix derivatives fetch (Cell 7, unchanged by insertion) ----
    print("\n[Part C] Patching Cell 7: DERIV_DAYS=180, windowed pagination")
    cells[7] = patch_cell_7_derivatives(cells[7])

    # ---- Part A: Update summary (now Cell 32, was Cell 31) ----
    print("\n[Part A] Patching Cell 32 (was 31): include lightgbm in summary table")
    cells[32] = patch_cell_31_summary(cells[32])

    # ---- Part D: Insert confidence analysis cell after summary (Cell 32) ----
    print("[Part D] Inserting new cell after Cell 32: Section 9b confidence analysis")
    conf_cell = create_confidence_analysis_cell()
    cells.insert(33, conf_cell)
    # After insertion: old Cell 33 (zip) -> Cell 34, old Cell 34 -> Cell 35

    # ---- Part E: Update zip/download cell (now Cell 34, was Cell 32) ----
    print("\n[Part E] Patching Cell 34 (was 32): include .pkl and .json in zip")
    cells[34] = patch_cell_32_zip(cells[34])

    nb['cells'] = cells
    print(f"\nPatched notebook: {len(cells)} cells")

    with open(NB_PATH, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\nSaved: {NB_PATH}")
    print("Done!")


if __name__ == '__main__':
    main()
