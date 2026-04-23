"""
bot/ — Extracted subsystems from RenaissanceTradingBot.

This package decomposes the god-class into focused modules:
  builder.py     — Component initialization (BotBuilder)
  signals.py     — Signal generation logic (SignalGenerator)
  decision.py    — Trading decision logic (DecisionMaker)
  cycle.py       — Main trading cycle (TradingCycleRunner)
  background.py  — Background async loops (BackgroundLoops)
"""
