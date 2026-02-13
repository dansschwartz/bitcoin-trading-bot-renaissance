"""
📊 INSTITUTIONAL DASHBOARD
==========================
Lightweight Flask UI to visualize bot consciousness and real-time metrics.
"""

from flask import Flask, render_template, jsonify
import threading
import logging
from datetime import datetime

class InstitutionalDashboard:
    def __init__(self, bot, host="0.0.0.0", port=5000):
        self.bot = bot
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return "🏛️ Renaissance Institutional Dashboard - LIVE"

        @self.app.route('/api/consciousness')
        def consciousness():
            """Returns the 'Inner Thoughts' of the bot for visualization."""
            # Pull metrics from the last processed cycle
            summary = self.bot.get_performance_summary()
            
            # Extract alpha decay alerts
            alpha_decay = {}
            if hasattr(self.bot, 'learning_engine'):
                alpha_decay = getattr(self.bot.learning_engine, '_latest_alpha_corrs', {})

            # Basis and NLP Metrics
            basis_metrics = {}
            if hasattr(self.bot, 'basis_engine'):
                basis_metrics = {"status": "ACTIVE"}

            # Meta-Strategy and Attribution
            strategy_metrics = {
                "current_mode": getattr(self.bot.strategy_selector, 'current_mode', 'TAKER'),
                "vpin": getattr(self.bot, 'last_vpin', 0.5)
            }

            # Advanced Risk Fortress Metrics
            risk_metrics = {}
            if hasattr(self.bot, 'risk_gateway'):
                risk_metrics = self.bot.risk_gateway.get_risk_metrics()
                # Inject VAE Anomaly score if available from last assessment
                risk_metrics['vae_anomaly_score'] = getattr(self.bot.risk_gateway, '_last_vae_loss', 0.0)

            attribution_summary = {}
            if hasattr(self.bot, 'attribution_engine'):
                attribution_summary = {"status": "READY"}
                
            return jsonify({
                "timestamp": datetime.now().isoformat(),
                "performance": summary,
                "alpha_decay": alpha_decay,
                "basis": basis_metrics,
                "strategy": strategy_metrics,
                "risk_fortress": risk_metrics,
                "attribution": attribution_summary,
                "status": "OPERATIONAL"
            })

        @self.app.route('/api/regime')
        def regime():
            regime_data = self.bot.regime_overlay.current_regime if hasattr(self.bot, 'regime_overlay') else {}
            return jsonify(regime_data)

        @self.app.route('/api/alerts')
        def alerts():
            """Aggregation of high-priority alerts for institutional traders."""
            alerts_list = []
            
            # 1. Regime transition alerts
            if hasattr(self.bot, 'regime_overlay'):
                regime = self.bot.regime_overlay.current_regime
                if regime:
                    if regime.get('volatility_regime') == 'high_volatility':
                        alerts_list.append({"level": "CRITICAL", "msg": "High Volatility Regime Detected - De-leveraging active."})
                    if regime.get('hmm_forecast') != regime.get('volatility_regime'):
                         alerts_list.append({"level": "WARNING", "msg": f"Predicted Regime Shift to {regime.get('hmm_forecast')}"})

            # 2. Alpha decay alerts
            if hasattr(self.bot, 'learning_engine'):
                corrs = self.bot.learning_engine._latest_alpha_corrs
                for k, v in corrs.items():
                    if abs(v) < 0.05:
                        alerts_list.append({"level": "INFO", "msg": f"Alpha Decay Alert: Signal '{k}' correlation dropped to {v:.4f}"})
            
            return jsonify({
                "timestamp": datetime.now().isoformat(),
                "alerts": alerts_list
            })

    def run(self):
        """Runs the dashboard in a background thread."""
        def run_flask():
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
            
        thread = threading.Thread(target=run_flask, daemon=True)
        thread.start()
        self.logger.info(f"📊 Institutional Dashboard started on {self.host}:{self.port}")
