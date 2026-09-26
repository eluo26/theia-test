"""Object localization for a laser-pointing turret.

Grok does scene understanding and query reasoning.
A local open-vocabulary detector tightens boxes when its weights are installed.
"""

from vision.index import build_catalog
from vision.query import locate

__all__ = ["build_catalog", "locate", "__version__"]

__version__ = "0.2.0"
