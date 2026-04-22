"""Allow running as: python -m arbitrage"""
import asyncio
from arbitrage.orchestrator import main

if __name__ == "__main__":
    asyncio.run(main())
