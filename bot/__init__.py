"""
bot/ — Extracted subsystems from RenaissanceTradingBot.

This package decomposes the god-class into focused modules:
  builder.py          — Component initialization (BotBuilder)
  signals.py          — Signal generation logic (SignalGenerator)
  decision.py         — Trading decision logic (DecisionMaker)
  data_collection.py  — Market data fetching & bar aggregation
  position_ops.py     — Position management, orders, state restore
  lifecycle.py        — Startup, shutdown, background loops, continuous trading
  adaptive.py         — Adaptive weights, attribution, Kelly calibration
"""
