#!/bin/bash
# =============================================================================
# Renaissance Trading Bot — Automated Weekly Model Retraining
# =============================================================================
#
# This script automates the full retraining cycle:
#   1. Sync project code to Google Drive
#   2. Open Colab notebook (you click "Run All" — the only manual step)
#   3. Pull trained models back from Google Drive
#   4. Deploy to the live bot
#
# SETUP (one-time):
#   1. Install Google Drive for Desktop: https://www.google.com/drive/download/
#   2. Sign in — it mounts at ~/Library/CloudStorage/GoogleDrive-<your-email>/
#   3. Run: ./retrain.sh setup
#
# WEEKLY USE:
#   ./retrain.sh sync      # Push latest code to Drive
#   ./retrain.sh train     # Opens Colab in browser — click "Run All"
#   ./retrain.sh deploy    # Pull new models from Drive, deploy to bot
#
#   OR do it all at once:
#   ./retrain.sh full      # sync → open Colab → wait → deploy
#
# =============================================================================

set -e

# ─── Configuration ───────────────────────────────────────────────────────────

BOT_DIR="$HOME/Downloads/bitcoin-trading-bot-renaissance"
MODELS_DIR="$BOT_DIR/models/trained"
SCRIPTS_DIR="$BOT_DIR/scripts/training"

# Google Drive paths (adjust email if needed)
GDRIVE_BASE="$HOME/Library/CloudStorage"
GDRIVE_DIR=""  # Auto-detected in find_gdrive()

DRIVE_PROJECT_DIR="renaissance-bot-training"
DRIVE_MODELS_DIR="renaissance-bot-training/trained_models"
DRIVE_NOTEBOOK="renaissance-bot-training/retrain_weekly_colab.ipynb"

# Colab URL (update after first upload)
COLAB_BASE_URL="https://colab.research.google.com/drive"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─── Helper Functions ────────────────────────────────────────────────────────

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

find_gdrive() {
    # Auto-detect Google Drive mount point
    if [ -d "$GDRIVE_BASE" ]; then
        GDRIVE_DIR=$(find "$GDRIVE_BASE" -maxdepth 1 -type d -name "GoogleDrive-*" 2>/dev/null | head -1)
        if [ -n "$GDRIVE_DIR" ]; then
            GDRIVE_DIR="$GDRIVE_DIR/My Drive"
            if [ -d "$GDRIVE_DIR" ]; then
                log_ok "Found Google Drive at: $GDRIVE_DIR"
                return 0
            fi
        fi
    fi
    
    # Try alternate locations
    for path in "$HOME/Google Drive/My Drive" "$HOME/GoogleDrive/My Drive"; do
        if [ -d "$path" ]; then
            GDRIVE_DIR="$path"
            log_ok "Found Google Drive at: $GDRIVE_DIR"
            return 0
        fi
    done
    
    log_error "Google Drive not found. Install Google Drive for Desktop first."
    log_info "Download: https://www.google.com/drive/download/"
    return 1
}

check_bot_dir() {
    if [ ! -f "$BOT_DIR/renaissance_trading_bot.py" ]; then
        log_error "Bot directory not found at $BOT_DIR"
        log_info "Edit BOT_DIR in this script to point to your project."
        exit 1
    fi
}

# ─── Commands ────────────────────────────────────────────────────────────────

cmd_setup() {
    echo ""
    echo "=========================================="
    echo "  ONE-TIME SETUP"
    echo "=========================================="
    echo ""
    
    check_bot_dir
    find_gdrive || exit 1
    
    # Create Drive directories
    log_info "Creating Google Drive folders..."
    mkdir -p "$GDRIVE_DIR/$DRIVE_PROJECT_DIR"
    mkdir -p "$GDRIVE_DIR/$DRIVE_MODELS_DIR"
    log_ok "Created $DRIVE_PROJECT_DIR/"
    
    # Sync project code to Drive (excluding heavy stuff)
    log_info "Syncing project code to Google Drive..."
    rsync -av --progress \
        --exclude='.venv/' \
        --exclude='__pycache__/' \
        --exclude='.git/' \
        --exclude='node_modules/' \
        --exclude='*.db' \
        --exclude='*.db-journal' \
        --exclude='data/training/*.csv' \
        --exclude='logs/' \
        "$BOT_DIR/" "$GDRIVE_DIR/$DRIVE_PROJECT_DIR/code/"
    
    log_ok "Project synced to Drive"
    
    # Copy the enhanced Colab notebook
    log_info "Creating enhanced Colab notebook..."
    create_colab_notebook
    log_ok "Notebook saved to Drive"
    
    echo ""
    log_ok "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Go to https://drive.google.com"
    echo "  2. Navigate to $DRIVE_PROJECT_DIR/"
    echo "  3. Double-click retrain_weekly_auto.ipynb to open in Colab"
    echo "  4. Runtime → Change runtime type → T4 GPU"
    echo "  5. Click 'Run All' (Ctrl+F9)"
    echo "  6. When done, run: ./retrain.sh deploy"
    echo ""
}

cmd_sync() {
    echo ""
    echo "=========================================="
    echo "  SYNCING CODE TO GOOGLE DRIVE"
    echo "=========================================="
    echo ""
    
    check_bot_dir
    find_gdrive || exit 1
    
    if [ ! -d "$GDRIVE_DIR/$DRIVE_PROJECT_DIR" ]; then
        log_error "Run './retrain.sh setup' first."
        exit 1
    fi
    
    log_info "Syncing latest code..."
    rsync -av --progress \
        --exclude='.venv/' \
        --exclude='__pycache__/' \
        --exclude='.git/' \
        --exclude='node_modules/' \
        --exclude='*.db' \
        --exclude='*.db-journal' \
        --exclude='data/training/*.csv' \
        --exclude='logs/' \
        --delete \
        "$BOT_DIR/" "$GDRIVE_DIR/$DRIVE_PROJECT_DIR/code/"
    
    log_ok "Code synced to Google Drive"
    echo ""
}

cmd_train() {
    echo ""
    echo "=========================================="
    echo "  OPENING COLAB FOR TRAINING"
    echo "=========================================="
    echo ""
    
    find_gdrive || exit 1
    
    NOTEBOOK_PATH="$GDRIVE_DIR/$DRIVE_PROJECT_DIR/retrain_weekly_auto.ipynb"
    
    if [ ! -f "$NOTEBOOK_PATH" ]; then
        log_warn "Notebook not found. Creating it..."
        create_colab_notebook
    fi
    
    log_info "Opening Colab in your browser..."
    log_info "When Colab opens:"
    echo "  1. Runtime → Change runtime type → T4 GPU (if not already set)"
    echo "  2. Runtime → Run all (Ctrl+F9)"
    echo "  3. Wait ~30-60 minutes for training to complete"
    echo "  4. The notebook auto-saves models to Google Drive"
    echo "  5. When done, come back here and run: ./retrain.sh deploy"
    echo ""
    
    # Open Google Drive in browser (user double-clicks notebook from there)
    open "https://drive.google.com/drive/search?q=retrain_weekly_auto.ipynb"
    
    echo ""
    log_info "Waiting for training to complete..."
    log_info "Press Enter after Colab finishes, or Ctrl+C to cancel."
    read -r
}

cmd_deploy() {
    echo ""
    echo "=========================================="
    echo "  DEPLOYING NEW MODELS"
    echo "=========================================="
    echo ""
    
    check_bot_dir
    find_gdrive || exit 1
    
    DRIVE_MODELS="$GDRIVE_DIR/$DRIVE_MODELS_DIR"
    
    if [ ! -d "$DRIVE_MODELS" ]; then
        log_error "No trained models found on Google Drive."
        log_info "Run training in Colab first: ./retrain.sh train"
        exit 1
    fi
    
    # Check if new models exist
    NEW_MODELS=$(find "$DRIVE_MODELS" -name "*.pth" -newer "$MODELS_DIR/best_quantum_transformer_model.pth" 2>/dev/null | wc -l)
    
    if [ "$NEW_MODELS" -eq 0 ]; then
        log_warn "No new models found (models on Drive are not newer than local)."
        log_info "Run training in Colab first, or use --force to deploy anyway."
        
        if [ "$1" != "--force" ]; then
            echo ""
            read -p "Deploy anyway? (y/N): " confirm
            if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
                exit 0
            fi
        fi
    fi
    
    # Backup current models
    BACKUP_DIR="$MODELS_DIR/../trained_backup_$(date +%Y%m%d_%H%M%S)"
    log_info "Backing up current models to $(basename $BACKUP_DIR)/"
    cp -r "$MODELS_DIR" "$BACKUP_DIR"
    log_ok "Backup saved"
    
    # Copy new models from Drive
    log_info "Copying new models from Google Drive..."
    cp "$DRIVE_MODELS"/*.pth "$MODELS_DIR/" 2>/dev/null || true
    cp "$DRIVE_MODELS"/training_metadata.json "$MODELS_DIR/" 2>/dev/null || true
    cp "$DRIVE_MODELS"/training_report_*.json "$MODELS_DIR/" 2>/dev/null || true
    
    # Count what we got
    MODEL_COUNT=$(ls "$MODELS_DIR"/*.pth 2>/dev/null | wc -l)
    log_ok "Deployed $MODEL_COUNT model files"
    
    # Show training metadata if available
    if [ -f "$MODELS_DIR/training_metadata.json" ]; then
        echo ""
        log_info "Training metadata:"
        python3 -c "
import json
with open('$MODELS_DIR/training_metadata.json') as f:
    meta = json.load(f)
for name, info in meta.items():
    acc = info.get('directional_accuracy', 'N/A')
    date = info.get('last_trained', 'unknown')
    if isinstance(acc, float):
        acc = f'{acc:.1%}'
    print(f'  {name:25s} | accuracy: {acc:>6s} | trained: {date[:10]}')
" 2>/dev/null || log_warn "Could not read training metadata"
    fi
    
    # Restart the bot
    echo ""
    read -p "Restart the bot now? (Y/n): " restart
    if [ "$restart" != "n" ] && [ "$restart" != "N" ]; then
        log_info "Stopping bot..."
        pkill -f "python.*main" 2>/dev/null || true
        sleep 2
        
        log_info "Starting bot..."
        cd "$BOT_DIR"
        .venv/bin/python3 renaissance_trading_bot.py > logs/bot_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        BOT_PID=$!
        sleep 3
        
        if kill -0 $BOT_PID 2>/dev/null; then
            log_ok "Bot restarted (PID: $BOT_PID)"
            log_info "Dashboard: http://localhost:8080"
        else
            log_error "Bot failed to start. Check logs/"
        fi
    fi
    
    echo ""
    log_ok "Deployment complete!"
    echo ""
    
    # Clean up old backups (keep last 4)
    BACKUP_COUNT=$(ls -d "$MODELS_DIR/../trained_backup_"* 2>/dev/null | wc -l)
    if [ "$BACKUP_COUNT" -gt 4 ]; then
        log_info "Cleaning old backups (keeping last 4)..."
        ls -dt "$MODELS_DIR/../trained_backup_"* | tail -n +5 | xargs rm -rf
        log_ok "Old backups cleaned"
    fi
}

cmd_full() {
    echo ""
    echo "=========================================="
    echo "  FULL RETRAINING CYCLE"
    echo "=========================================="
    echo ""
    
    cmd_sync
    cmd_train
    cmd_deploy
    
    echo ""
    log_ok "Full retraining cycle complete!"
    echo ""
}

cmd_status() {
    echo ""
    echo "=========================================="
    echo "  MODEL STATUS"
    echo "=========================================="
    echo ""
    
    check_bot_dir
    
    if [ -f "$MODELS_DIR/training_metadata.json" ]; then
        python3 -c "
import json
from datetime import datetime, timezone

with open('$MODELS_DIR/training_metadata.json') as f:
    meta = json.load(f)

print(f'  {\"Model\":25s} | {\"Accuracy\":>8s} | {\"Trained\":>12s} | {\"Age\":>8s} | Status')
print(f'  {\"-\"*25} | {\"-\"*8} | {\"-\"*12} | {\"-\"*8} | ------')

for name, info in meta.items():
    acc = info.get('directional_accuracy', 0)
    date_str = info.get('last_trained', '')
    
    if date_str:
        trained = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        age = (datetime.now(timezone.utc) - trained).days
        age_str = f'{age}d'
        status = '✅ OK' if age <= 7 else '⚠️  STALE' if age <= 14 else '❌ OLD'
    else:
        age_str = '?'
        status = '❓ Unknown'
    
    acc_str = f'{acc:.1%}' if isinstance(acc, float) else 'N/A'
    date_short = date_str[:10] if date_str else 'never'
    
    print(f'  {name:25s} | {acc_str:>8s} | {date_short:>12s} | {age_str:>8s} | {status}')
" 2>/dev/null
    else
        log_warn "No training metadata found. Models may not have been trained with the new pipeline."
        echo ""
        log_info "Model files in $MODELS_DIR:"
        ls -la "$MODELS_DIR"/*.pth 2>/dev/null | awk '{print "  " $6, $7, $8, $9}'
    fi
    
    echo ""
}

# ─── Colab Notebook Generator ────────────────────────────────────────────────

create_colab_notebook() {
    find_gdrive || return 1
    
    NOTEBOOK_PATH="$GDRIVE_DIR/$DRIVE_PROJECT_DIR/retrain_weekly_auto.ipynb"
    
    cat > "$NOTEBOOK_PATH" << 'NOTEBOOK_EOF'
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 🏛️ Renaissance Bot — Automated Weekly Retraining\n",
        "\n",
        "**One-click training pipeline.** Just click `Runtime → Run All` (Ctrl+F9).\n",
        "\n",
        "What this does:\n",
        "1. Mounts your Google Drive\n",
        "2. Copies latest code from Drive\n",
        "3. Installs dependencies\n",
        "4. Downloads 30 days of market data\n",
        "5. Trains all 7 models with GPU acceleration\n",
        "6. Saves trained models back to Google Drive\n",
        "7. Generates a training report\n",
        "\n",
        "**Prerequisites:** Runtime → Change runtime type → T4 GPU"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# ============================================================\n",
        "# CELL 1: Setup — Mount Drive & Copy Code\n",
        "# ============================================================\n",
        "from google.colab import drive\n",
        "import os, shutil\n",
        "\n",
        "# Mount Google Drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Configuration\n",
        "DRIVE_PROJECT = '/content/drive/MyDrive/renaissance-bot-training'\n",
        "DRIVE_CODE = f'{DRIVE_PROJECT}/code'\n",
        "DRIVE_MODELS = f'{DRIVE_PROJECT}/trained_models'\n",
        "WORK_DIR = '/content/renaissance-bot'\n",
        "\n",
        "# Copy code from Drive to local (faster I/O)\n",
        "if os.path.exists(WORK_DIR):\n",
        "    shutil.rmtree(WORK_DIR)\n",
        "shutil.copytree(DRIVE_CODE, WORK_DIR)\n",
        "os.chdir(WORK_DIR)\n",
        "\n",
        "# Create output directory on Drive\n",
        "os.makedirs(DRIVE_MODELS, exist_ok=True)\n",
        "\n",
        "print(f'Working directory: {os.getcwd()}')\n",
        "print(f'Files: {len(os.listdir(\".\"))}')\n",
        "print('Setup complete ✅')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# ============================================================\n",
        "# CELL 2: Install Dependencies\n",
        "# ============================================================\n",
        "!pip install -q torch numpy pandas scikit-learn hmmlearn\n",
        "\n",
        "import torch\n",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "if device == 'cuda':\n",
        "    gpu_name = torch.cuda.get_device_name(0)\n",
        "    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9\n",
        "    print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB) ✅')\n",
        "else:\n",
        "    print('⚠️  No GPU detected! Training will be slow.')\n",
        "    print('Go to Runtime → Change runtime type → T4 GPU')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# ============================================================\n",
        "# CELL 3: Download Fresh Market Data\n",
        "# ============================================================\n",
        "!python -m scripts.training.fetch_training_data --days 30\n",
        "\n",
        "# Show what we got\n",
        "import glob\n",
        "csvs = glob.glob('data/training/*.csv')\n",
        "for f in sorted(csvs):\n",
        "    lines = sum(1 for _ in open(f)) - 1\n",
        "    print(f'  {os.path.basename(f):15s} {lines:,} candles')\n",
        "print(f'Data download complete ✅')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# ============================================================\n",
        "# CELL 4: Train All 7 Models (100 epochs, early stopping)\n",
        "# ============================================================\n",
        "# This is the main training cell. Takes ~30-60 min on T4 GPU.\n",
        "import time\n",
        "start = time.time()\n",
        "\n",
        "!python -m scripts.training.train_all --days 30 --epochs 100\n",
        "\n",
        "elapsed = (time.time() - start) / 60\n",
        "print(f'\\nTotal training time: {elapsed:.1f} minutes')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# ============================================================\n",
        "# CELL 5: Display Results & Training Report\n",
        "# ============================================================\n",
        "import json, glob\n",
        "\n",
        "# Load training metadata\n",
        "meta_path = 'models/trained/training_metadata.json'\n",
        "if os.path.exists(meta_path):\n",
        "    with open(meta_path) as f:\n",
        "        meta = json.load(f)\n",
        "    \n",
        "    print('=' * 65)\n",
        "    print('  TRAINING RESULTS')\n",
        "    print('=' * 65)\n",
        "    print(f'  {\"Model\":25s} | {\"Val Acc\":>8s} | {\"Test Acc\":>8s} | Status')\n",
        "    print(f'  {\"-\"*25} | {\"-\"*8} | {\"-\"*8} | ------')\n",
        "    \n",
        "    all_good = True\n",
        "    for name, info in meta.items():\n",
        "        acc = info.get('directional_accuracy', 0)\n",
        "        val_acc = info.get('validation_accuracy', acc)\n",
        "        if isinstance(acc, float) and acc > 0:\n",
        "            status = '✅' if acc > 0.52 else '⚠️' if acc > 0.50 else '❌'\n",
        "            if acc <= 0.50:\n",
        "                all_good = False\n",
        "            print(f'  {name:25s} | {val_acc:>7.1%} | {acc:>7.1%} | {status}')\n",
        "        else:\n",
        "            print(f'  {name:25s} | {\"N/A\":>8s} | {\"N/A\":>8s} | ✅ (VAE)')\n",
        "    \n",
        "    print()\n",
        "    if all_good:\n",
        "        print('  All models above chance level ✅')\n",
        "    else:\n",
        "        print('  ⚠️  Some models below 50% — check training data')\n",
        "else:\n",
        "    print('No training metadata found. Check training output above for errors.')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# ============================================================\n",
        "# CELL 6: Save Models to Google Drive (auto-deploy)\n",
        "# ============================================================\n",
        "import shutil, glob\n",
        "\n",
        "# Copy all trained model files to Drive\n",
        "model_files = glob.glob('models/trained/*.pth')\n",
        "meta_files = glob.glob('models/trained/*.json')\n",
        "\n",
        "for f in model_files + meta_files:\n",
        "    dest = os.path.join(DRIVE_MODELS, os.path.basename(f))\n",
        "    shutil.copy2(f, dest)\n",
        "    size_mb = os.path.getsize(f) / 1e6\n",
        "    print(f'  Saved: {os.path.basename(f):45s} ({size_mb:.1f} MB)')\n",
        "\n",
        "print(f'\\n{len(model_files)} model files + {len(meta_files)} metadata files saved to Google Drive')\n",
        "print(f'Location: {DRIVE_MODELS}')\n",
        "print()\n",
        "print('=' * 65)\n",
        "print('  DONE! Models are on Google Drive.')\n",
        "print('  On your Mac, run: ./retrain.sh deploy')\n",
        "print('=' * 65)"
      ],
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
NOTEBOOK_EOF

    log_ok "Colab notebook saved to Google Drive"
}

# ─── Main ────────────────────────────────────────────────────────────────────

show_help() {
    echo ""
    echo "Renaissance Trading Bot — Model Retraining Tool"
    echo ""
    echo "Usage: ./retrain.sh <command>"
    echo ""
    echo "Commands:"
    echo "  setup    One-time setup: create Drive folders, sync code, create notebook"
    echo "  sync     Push latest code to Google Drive"
    echo "  train    Open Colab in browser for GPU training"
    echo "  deploy   Pull trained models from Drive and restart bot"
    echo "  full     Run the complete cycle: sync → train → deploy"
    echo "  status   Show current model status and staleness"
    echo "  help     Show this help message"
    echo ""
    echo "Weekly workflow:"
    echo "  ./retrain.sh full"
    echo ""
}

case "${1:-help}" in
    setup)  cmd_setup ;;
    sync)   cmd_sync ;;
    train)  cmd_train ;;
    deploy) cmd_deploy "$2" ;;
    full)   cmd_full ;;
    status) cmd_status ;;
    help)   show_help ;;
    *)      log_error "Unknown command: $1"; show_help; exit 1 ;;
esac
