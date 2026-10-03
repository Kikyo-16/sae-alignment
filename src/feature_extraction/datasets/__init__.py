from __future__ import annotations

"""
Dataset handlers for feature extraction.

All dataset modules under this package are auto-imported, and each module
registers itself with the central registry on import.
"""

from importlib import import_module
from pkgutil import iter_modules


def _auto_import_dataset_modules() -> None:
    # Auto-discover dataset modules in this package so new datasets only need
    # one new file under feature_extraction/datasets/.
    for mod in iter_modules(__path__):
        name = str(mod.name)
        if name.startswith("_"):
            continue
        import_module(f"{__name__}.{name}")


_auto_import_dataset_modules()

