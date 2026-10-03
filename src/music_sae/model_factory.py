from __future__ import annotations

import typing as tp
import torch


class ModelFactory:
    def __init__(self, model_config: dict[str, tp.Any]) -> None:
        self.model_dict: dict[str, int] = model_config["feature_dims"]
        model_cfgs: dict[str, tp.Any] = model_config.get("models", {})
        self.models: dict[str, tp.Any] = {}

        for base_name, cfg in model_cfgs.items():
            print(base_name)
            if cfg is None:
                self.models[base_name] = None
                continue

            model_cls = cfg.get("model")
            ckpt_path = cfg.get("path", None)

            if ckpt_path is not None:
                model = model_cls.load_from_checkpoint(ckpt_path, strict=False)
            else:
                model = model_cls(**cfg.get("args"))
            model.eval()
            if hasattr(model, "autoDataStats"):
                model.autoDataStats = False
            self.models[base_name] = model

    def autoDataStats(self, flag):
        for k in self.models:
            if self.models[k] is not None and hasattr(self.models[k], "autoDataStats"):
                self.models[k].autoDataStats = flag

    def to(self, device):
        for model_name, model in self.models.items():
            if model is None:
                continue
            model.to(device)
            if hasattr(model, "on_fit_start") and callable(model.on_fit_start):
                model.on_fit_start()

    def _parse_model_name(self, name: str) -> tuple[str, list[int]]:
        parts = name.split("_")
        if len(parts) < 3 or parts[1] != "layer":
            raise ValueError(f"Bad feature key: {name} (expected '<base>_layer_<id>[_...]')")
        base = parts[0]
        try:
            layers = [int(p) for p in parts[2:]]
        except Exception as e:
            raise ValueError(f"Bad layer ids in key: {name}") from e
        if not layers:
            raise ValueError(f"No layer ids found in key: {name}")
        return base, layers

    def _rebuild_full_name(self, model_name: str, layer_id: int) -> str:
        return f"{model_name}_layer_{layer_id}"

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor], t: int | None = None) -> dict[str, torch.Tensor]:
        grouped: dict[str, dict[str, tp.Any]] = {}
        for full_name, x in batch.items():
            base, layer_indices = self._parse_model_name(full_name)
            if base not in grouped:
                grouped[base] = {"layers": set(layer_indices), "x": x.float()}
            else:
                grouped[base]["layers"].update(layer_indices)
                if grouped[base]["x"] is not x and grouped[base]["x"].shape != x.shape:
                    raise ValueError(
                        f"Multiple tensors for base='{base}' with different shapes: "
                        f"{tuple(grouped[base]['x'].shape)} vs {tuple(x.shape)}"
                    )

        out: dict[str, torch.Tensor] = {}
        for base, pack in grouped.items():
            layers = sorted(pack["layers"])
            x = pack["x"]
            model = self.models.get(base, None)

            if model is None:
                for layer_id in layers:
                    out[self._rebuild_full_name(base, layer_id)] = x.transpose(-1, -2)
            else:
                model_out = model.sample_feature(basename=base, x=x, layers=layers, t=t)
                if not isinstance(model_out, dict):
                    raise TypeError(f"{base}.sample_feature must return dict, got {type(model_out)}")
                out.update(model_out)

        return out
