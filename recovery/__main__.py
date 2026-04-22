"""Allow ``python -m recovery`` to launch the watchdog by default."""
from recovery.watchdog import _main

if __name__ == "__main__":
    _main()
