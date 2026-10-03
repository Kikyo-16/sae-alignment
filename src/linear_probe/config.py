from __future__ import annotations

import typing as tp


def _make_probe_config(feature_tag: str, d_model: int) -> dict[str, tp.Any]:
    return {
        "feature_tag": feature_tag,
        "d_model": d_model,
    }


probe_config: dict[str, dict[str, tp.Any]] = {
    "muq-2": _make_probe_config("muq_layer", 1024),
    "muq-3": _make_probe_config("muq_layer", 1024),
    "muq-4": _make_probe_config("muq_layer", 1024),
    "muq-5": _make_probe_config("muq_layer", 1024),
    "muq-6": _make_probe_config("muq_layer", 1024),
    "muq-8": _make_probe_config("muq_layer", 1024),
    "muq-10": _make_probe_config("muq_layer", 1024),
    "muq-0": _make_probe_config("muq_layer", 1024),
    "musicfm-2": _make_probe_config("musicfm_layer", 1024),
    "musicfm-3": _make_probe_config("musicfm_layer", 1024),
    "musicfm-4": _make_probe_config("musicfm_layer", 1024),
    "musicfm-5": _make_probe_config("musicfm_layer", 1024),
    "musicfm-6": _make_probe_config("musicfm_layer", 1024),
    "musicfm-8": _make_probe_config("musicfm_layer", 1024),
    "musicfm-10": _make_probe_config("musicfm_layer", 1024),
    "musicfm-0": _make_probe_config("musicfm_layer", 1024),
}
