from __future__ import annotations

import typing as tp


def _make_model_config(data_tag: str, layer_id: int, d_model: int) -> dict[str, tp.Any]:
    return {
        "data_tag": f"{data_tag}_{layer_id}",
        "feature_dims": {f"{data_tag}_{layer_id}": d_model},
        "models": {"feature": None},  # features pre-extracted to H5, passthrough
    }


aug_order_1 = ["ps-2", "ps-1", "orig", "ps+1", "ps+2"]


def _make_exp(model_config: dict[str, tp.Any], **kwargs: tp.Any) -> dict[str, tp.Any]:
    base = {
        "aug": True,
        "share_mask": True,
        "n_saes": 3,
        "mlp_ratio": 4,
        "aug_order": aug_order_1,
        "norm_every": 1,
        "model_config": model_config,
        "norm_stats_path": None,
    }
    base.update(kwargs)
    return base


def build_exp_config(feature: str, layer_id: int, n_saes: int, d_model: int = 1024) -> dict[str, tp.Any]:
    """exp_config for ``<feature>-<layer_id>_<n_saes>-1`` (same as the entries below).

    feature: foundation model name, e.g. "muq" or "musicfm".
    n_saes:  3 -> o-SAE (aligned multi-SAE), 1 -> s-SAE (standard SAE ablation).
    """
    return _make_exp(_make_model_config(f"{feature}_layer", layer_id, d_model),
                     n_saes=n_saes, aug_order=aug_order_1)


exp_config: dict[str, dict[str, tp.Any]] = {
    "musicfm-10_1-1": _make_exp(_make_model_config("musicfm_layer", 10, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-8_1-1": _make_exp(_make_model_config("musicfm_layer", 8, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-6_1-1": _make_exp(_make_model_config("musicfm_layer", 6, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-5_1-1": _make_exp(_make_model_config("musicfm_layer", 5, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-4_1-1": _make_exp(_make_model_config("musicfm_layer", 4, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-3_1-1": _make_exp(_make_model_config("musicfm_layer", 3, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-2_1-1": _make_exp(_make_model_config("musicfm_layer", 2, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-0_1-1": _make_exp(_make_model_config("musicfm_layer", 0, 1024),
                            n_saes=1, aug_order=aug_order_1),


    "musicfm-10_3-1": _make_exp(_make_model_config("musicfm_layer", 10, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-8_3-1": _make_exp(_make_model_config("musicfm_layer", 8, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-6_3-1": _make_exp(_make_model_config("musicfm_layer", 6, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-5_3-1": _make_exp(_make_model_config("musicfm_layer", 5, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-4_3-1": _make_exp(_make_model_config("musicfm_layer", 4, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-3_3-1": _make_exp(_make_model_config("musicfm_layer", 3, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-2_3-1": _make_exp(_make_model_config("musicfm_layer", 2, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-0_3-1": _make_exp(_make_model_config("musicfm_layer", 0, 1024),
                            n_saes=3, aug_order=aug_order_1),


    "muq-10_3-1": _make_exp(_make_model_config("muq_layer", 10, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-8_3-1": _make_exp(_make_model_config("muq_layer", 8, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-6_3-1": _make_exp(_make_model_config("muq_layer", 6, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-5_3-1": _make_exp(_make_model_config("muq_layer", 5, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-4_3-1": _make_exp(_make_model_config("muq_layer", 4, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-3_3-1": _make_exp(_make_model_config("muq_layer", 3, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-2_3-1": _make_exp(_make_model_config("muq_layer", 2, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-0_3-1": _make_exp(_make_model_config("muq_layer", 0, 1024),
                            n_saes=3, aug_order=aug_order_1),


    "muq-10_1-1": _make_exp(_make_model_config("muq_layer", 10, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-8_1-1": _make_exp(_make_model_config("muq_layer", 8, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-6_1-1": _make_exp(_make_model_config("muq_layer", 6, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-5_1-1": _make_exp(_make_model_config("muq_layer", 5, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-4_1-1": _make_exp(_make_model_config("muq_layer", 4, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-3_1-1": _make_exp(_make_model_config("muq_layer", 3, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-2_1-1": _make_exp(_make_model_config("muq_layer", 2, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-0_1-1": _make_exp(_make_model_config("muq_layer", 0, 1024),
                            n_saes=1, aug_order=aug_order_1),


}
