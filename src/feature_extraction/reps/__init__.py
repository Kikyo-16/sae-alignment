"""
Rep (feature representation) handlers for `pack_split_dataset`.

All rep modules under this package are auto-imported, and each module registers
its handler(s) with the central registry on import — so adding a new rep only
needs one new file under feature_extraction/reps/ (no edits here).

The paper uses muq_layer (MuQ) and musicfm_layer (MusicFM). A rep module whose
optional third-party dependency is not installed is skipped with a notice; it
only fails if that rep is actually requested.
"""
from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules


def _auto_import_rep_modules() -> None:
    for mod in iter_modules(__path__):
        name = str(mod.name)
        if name.startswith("_"):
            continue
        try:
            import_module(f"{__name__}.{name}")
        except ImportError as exc:
            print(f"[reps] skipped rep module '{name}' (missing dependency: {exc})")


_auto_import_rep_modules()
