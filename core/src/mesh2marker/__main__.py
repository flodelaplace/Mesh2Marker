"""Entry point for ``python -m mesh2marker``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
