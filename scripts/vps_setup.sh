#!/usr/bin/env bash
# ==============================================================================
# Renaissance Trading Bot — VPS Setup Script
#
# Usage:
#   1. Create a Ubuntu 24.04 VPS (DigitalOcean, Hetzner, Linode, etc.)
#   2. scp this script to the server:
#        scp scripts/vps_setup.sh root@YOUR_SERVER_IP:/root/
#   3. SSH in and run:
#        ssh root@YOUR_SERVER_IP
#        chmod +x vps_setup.sh && ./vps_setup.sh
#   4. After setup, copy your config:
#        scp config/config.json botuser@YOUR_SERVER_IP:/home/botuser/bitcoin-trading-bot-renaissance/config/
#   5. Start the bot:
#        ssh botuser@YOUR_SERVER_IP
#        sudo systemctl start renaissance-bot
#        sudo systemctl start renaissance-dashboard
#
# What this script does:
#   - Installs Python 3.11, git, tmux, sqlite3
#   - Creates 'botuser' user (if not exists)
#   - Clones the repo
#   - Sets up Python venv and installs dependencies
#   - Creates systemd services for bot + dashboard (auto-restart on crash/reboot)
#   - Sets up log rotation
#   - Opens firewall for SSH only (dashboard accessed via SSH tunnel)
# ==============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/dansschwartz/bitcoin-trading-bot-renaissance.git"
BOT_USER="botuser"
BOT_HOME="/home/${BOT_USER}"
APP_DIR="${BOT_HOME}/bitcoin-trading-bot-renaissance"
PYTHON="python3.11"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Pre-checks ────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root. Use: sudo ./vps_setup.sh"
fi

info "Starting VPS setup for Renaissance Trading Bot..."

# ── Step 1: System packages ──────────────────────────────────────────────────
info "Updating system packages..."
apt update -qq && apt upgrade -y -qq

info "Installing dependencies..."
apt install -y -qq \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip git tmux sqlite3 curl ufw \
    build-essential libffi-dev libssl-dev

# ── Step 2: Create bot user ──────────────────────────────────────────────────
if id "${BOT_USER}" &>/dev/null; then
    info "User '${BOT_USER}' already exists."
else
    info "Creating user '${BOT_USER}'..."
    adduser --disabled-password --gecos "Renaissance Bot" "${BOT_USER}"
    usermod -aG sudo "${BOT_USER}"
    # Allow sudo without password for systemctl commands only
    echo "${BOT_USER} ALL=(ALL) NOPASSWD: /usr/bin/systemctl" > /etc/sudoers.d/${BOT_USER}
    chmod 440 /etc/sudoers.d/${BOT_USER}
fi

# Copy SSH keys so botuser can be accessed via SSH
if [[ -f /root/.ssh/authorized_keys ]]; then
    mkdir -p "${BOT_HOME}/.ssh"
    cp /root/.ssh/authorized_keys "${BOT_HOME}/.ssh/"
    chown -R "${BOT_USER}:${BOT_USER}" "${BOT_HOME}/.ssh"
    chmod 700 "${BOT_HOME}/.ssh"
    chmod 600 "${BOT_HOME}/.ssh/authorized_keys"
    info "SSH keys copied to ${BOT_USER}."
fi

# ── Step 3: Clone repo ──────────────────────────────────────────────────────
if [[ -d "${APP_DIR}" ]]; then
    info "Repo already exists at ${APP_DIR}, pulling latest..."
    su - "${BOT_USER}" -c "cd ${APP_DIR} && git pull"
else
    info "Cloning repository..."
    su - "${BOT_USER}" -c "git clone ${REPO_URL} ${APP_DIR}"
fi

# ── Step 4: Python venv + dependencies ───────────────────────────────────────
info "Setting up Python virtual environment..."
su - "${BOT_USER}" -c "
    cd ${APP_DIR}
    ${PYTHON} -m venv .venv
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -r requirements.txt -q
"
info "Python dependencies installed."

# ── Step 5: Create data directories ─────────────────────────────────────────
su - "${BOT_USER}" -c "
    mkdir -p ${APP_DIR}/data
    mkdir -p ${APP_DIR}/data/heartbeats
    mkdir -p ${APP_DIR}/data/research_sessions
    mkdir -p ${APP_DIR}/logs
    mkdir -p ${APP_DIR}/reports
"

# ── Step 6: Systemd service — Bot ────────────────────────────────────────────
info "Creating systemd service for the trading bot..."
cat > /etc/systemd/system/renaissance-bot.service << UNIT
[Unit]
Description=Renaissance Trading Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=${BOT_USER}
Group=${BOT_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/python run_renaissance_bot.py
Restart=always
RestartSec=15
StartLimitIntervalSec=300
StartLimitBurst=5

# Logging
StandardOutput=append:${APP_DIR}/logs/bot.log
StandardError=append:${APP_DIR}/logs/bot_error.log

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT

# ── Step 7: Systemd service — Dashboard ──────────────────────────────────────
info "Creating systemd service for the dashboard..."
cat > /etc/systemd/system/renaissance-dashboard.service << UNIT
[Unit]
Description=Renaissance Trading Bot Dashboard
After=network.target renaissance-bot.service

[Service]
Type=simple
User=${BOT_USER}
Group=${BOT_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/python -m dashboard.server
Restart=always
RestartSec=10

# Logging
StandardOutput=append:${APP_DIR}/logs/dashboard.log
StandardError=append:${APP_DIR}/logs/dashboard_error.log

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT

# ── Step 8: Enable services ──────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable renaissance-bot.service
systemctl enable renaissance-dashboard.service
info "Services enabled (will auto-start on reboot)."

# ── Step 9: Log rotation ────────────────────────────────────────────────────
info "Setting up log rotation..."
cat > /etc/logrotate.d/renaissance-bot << LOGROTATE
${APP_DIR}/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
LOGROTATE

# ── Step 10: Firewall ───────────────────────────────────────────────────────
info "Configuring firewall (SSH only, dashboard via tunnel)..."
ufw --force reset > /dev/null 2>&1
ufw default deny incoming > /dev/null
ufw default allow outgoing > /dev/null
ufw allow ssh > /dev/null
ufw --force enable > /dev/null
info "Firewall active: SSH allowed, all other inbound blocked."

# ── Step 11: Swap file (for 1GB RAM servers) ─────────────────────────────────
if [[ ! -f /swapfile ]]; then
    info "Creating 2GB swap file (helps on low-memory servers)..."
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile > /dev/null
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    info "Swap enabled."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================================================="
echo -e "${GREEN}  VPS SETUP COMPLETE${NC}"
echo "=============================================================================="
echo ""
echo "  Next steps:"
echo ""
echo "  1. Copy your config file from your laptop:"
echo "     scp config/config.json ${BOT_USER}@YOUR_SERVER_IP:${APP_DIR}/config/"
echo ""
echo "  2. Start the bot:"
echo "     sudo systemctl start renaissance-bot"
echo "     sudo systemctl start renaissance-dashboard"
echo ""
echo "  3. Check status:"
echo "     sudo systemctl status renaissance-bot"
echo "     sudo journalctl -u renaissance-bot -f    (live logs)"
echo ""
echo "  4. Access dashboard from your laptop:"
echo "     ssh -L 8080:localhost:8080 ${BOT_USER}@YOUR_SERVER_IP"
echo "     Then open http://localhost:8080 in your browser"
echo ""
echo "  5. SSH in as botuser:"
echo "     ssh ${BOT_USER}@YOUR_SERVER_IP"
echo ""
echo "  Useful commands:"
echo "     sudo systemctl restart renaissance-bot       # restart bot"
echo "     sudo systemctl stop renaissance-bot          # stop bot"
echo "     sudo systemctl status renaissance-bot        # check if running"
echo "     tail -f ${APP_DIR}/logs/bot.log              # live bot logs"
echo "     tail -f ${APP_DIR}/logs/dashboard.log        # live dashboard logs"
echo ""
echo "=============================================================================="
