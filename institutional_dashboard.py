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
            return jsonify({
                "timestamp": datetime.now().isoformat(),
                "performance": summary,
                "status": "OPERATIONAL"
            })

        @self.app.route('/api/regime')
        def regime():
            regime_data = self.bot.regime_overlay.get_current_regime() if hasattr(self.bot, 'regime_overlay') else {}
            return jsonify(regime_data)

    def run(self):
        """Runs the dashboard in a background thread."""
        def run_flask():
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
            
        thread = threading.Thread(target=run_flask, daemon=True)
        thread.start()
        self.logger.info(f"📊 Institutional Dashboard started on {self.host}:{self.port}")
