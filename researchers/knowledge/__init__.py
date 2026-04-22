"""
Executive Research Council — Executable Knowledge Library
SAND-inspired columnar architecture with bit-vector indexing.

Usage:
    from knowledge.registry import KB
    from knowledge.atoms import *  # registers all atoms

    # Query by any dimension:
    atoms = KB.query(pairs=KB.BTC, regimes=KB.TRENDING)

    # Execute a formula:
    result = KB.execute("math.kelly_optimal", p=0.54, b=1.2)

    # Full docs:
    print(KB.explain("math.kelly_optimal"))

    # Shared utilities:
    from knowledge.shared.data_loader import load_pair_csv
    from knowledge.shared.queries import weekly_performance
    from knowledge.shared.dead_ends import is_dead_end
"""
__version__ = "1.0.0"
