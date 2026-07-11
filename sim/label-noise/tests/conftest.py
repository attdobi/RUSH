"""Make `import labelsim` resolve when pytest runs from the repo root.

The package lives in sim/label-noise/ (a dash, so not itself importable);
inserting that directory ahead of everything else exposes `labelsim`.
"""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[1]  # sim/label-noise
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
