"""
Train 6 Oracle MLP Models
==========================
Trains the 6 MLP classifiers (one per backward/forward window combination)
using the processed training CSV.

Architecture: 36 → 128 → 64 → 32 → 3 (ReLU + Softmax)
~15K parameters per model.

Uses stratified random sampling (SRS) for class imbalance.
Trained on data before 2025-01-01, tested on data after.

Usage:
    python3 scripts/oracle/train_models.py

Requires: tensorflow, keras, scikit-learn, pandas, numpy
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'oracle',
    'raw_data_4_hour_train_test_data.csv'
)
# Also check the original location
ALT_DATA_PATH = os.path.expanduser(
    '~/Downloads/22953377/CryptoTrading/processed_data/'
    'raw_data_4_hour_train_test_data.csv'
)
SCALER_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'oracle', 'oracle_scaler.pkl'
)
MODEL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'oracle'
)

FEATURE_COLS = [
    'Z_score', 'RSI', 'boll', 'ULTOSC', 'pct_change', 'zsVol',
    'PR_MA_Ratio_short', 'MA_Ratio_short', 'MA_Ratio', 'PR_MA_Ratio',
    'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3WHITESOLDIERS',
    'CDLABANDONEDBABY', 'CDLBELTHOLD', 'CDLCOUNTERATTACK',
    'CDLDARKCLOUDCOVER', 'CDLDRAGONFLYDOJI', 'CDLENGULFING',
    'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGRAVESTONEDOJI',
    'CDLHANGINGMAN', 'CDLHARAMICROSS', 'CDLINVERTEDHAMMER',
    'CDLMARUBOZU', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
    'CDLPIERCING', 'CDLRISEFALL3METHODS', 'CDLSHOOTINGSTAR',
    'CDLSPINNINGTOP', 'CDLUPSIDEGAP2CROWS',
    'DayOfWeek', 'Month', 'Hourly',
]

# The 6 model configurations (backward_window, forward_window)
MODEL_CONFIGS = [
    {'bw': 2, 'fw': 1},
    {'bw': 2, 'fw': 2},
    {'bw': 3, 'fw': 2},
    {'bw': 4, 'fw': 2},
    {'bw': 5, 'fw': 1},
    {'bw': 5, 'fw': 2},
]

TRAIN_CUTOFF = '2025-01-01'
EPOCHS = 50
BATCH_SIZE = 512


def build_model(n_features: int = 36, n_classes: int = 3):
    """Build the MLP classifier (~15K parameters)."""
    import keras
    from keras import layers

    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    data_path = DATA_PATH if os.path.exists(DATA_PATH) else ALT_DATA_PATH
    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at {DATA_PATH} or {ALT_DATA_PATH}")
        print("Copy raw_data_4_hour_train_test_data.csv to data/oracle/")
        sys.exit(1)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows")

    # Clean features
    features = df[FEATURE_COLS].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Keep rows where features are valid
    valid_mask = features.notna().all(axis=1)
    df = df[valid_mask].copy()
    features = features[valid_mask].copy()
    print(f"After cleaning: {len(df):,} rows")

    # Load or fit scaler
    if os.path.exists(SCALER_PATH):
        print(f"Loading existing scaler from {SCALER_PATH}")
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print("Fitting new scaler...")
        scaler = StandardScaler()
        scaler.fit(features.values)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {SCALER_PATH}")

    # Scale features
    X_all = scaler.transform(features.values)

    # Split by date
    dates = pd.to_datetime(df['Date'])
    train_mask = dates < TRAIN_CUTOFF
    test_mask = dates >= TRAIN_CUTOFF

    X_train = X_all[train_mask]
    X_test = X_all[test_mask]

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train each model
    for cfg in MODEL_CONFIGS:
        bw, fw = cfg['bw'], cfg['fw']
        label_col = f'lab_{bw}_{fw}'
        model_name = f'model_final_{bw}_{fw}'

        if label_col not in df.columns:
            print(f"WARNING: Label column {label_col} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Training {model_name} (bw={bw}, fw={fw})")
        print(f"{'='*60}")

        # Get labels — map {-1, 0, 1} → {0, 1, 2} for sparse categorical
        y_all = df[label_col].values
        y_train = y_all[train_mask] + 1  # -1→0, 0→1, 1→2
        y_test = y_all[test_mask] + 1

        # Class weights for imbalanced data (SRS approach)
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        print(f"Class weights: {class_weight}")

        # Build and train
        model = build_model()
        print(f"Parameters: {model.count_params():,}")

        import keras
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True,
                    monitor='val_accuracy',
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=5, min_lr=1e-6,
                    monitor='val_accuracy',
                ),
            ],
            verbose=1,
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        print(f"\n{model_name}: Train acc={train_acc:.4f} | Test acc={test_acc:.4f}")

        # Dummy baseline
        from collections import Counter
        most_common = Counter(y_test).most_common(1)[0]
        dummy_acc = most_common[1] / len(y_test)
        print(f"Dummy baseline: {dummy_acc:.4f} (always predict class {most_common[0]})")

        # Save
        save_path = os.path.join(MODEL_DIR, f'{model_name}.h5')
        model.save(save_path)
        print(f"Saved to {save_path}")

    print("\n" + "=" * 60)
    print("All models trained!")
    print("=" * 60)
    for cfg in MODEL_CONFIGS:
        path = os.path.join(MODEL_DIR, f'model_final_{cfg["bw"]}_{cfg["fw"]}.h5')
        exists = os.path.exists(path)
        print(f"  {path}: {'OK' if exists else 'MISSING'}")


if __name__ == '__main__':
    main()
