"""NeSymReS layer registry and freeze utilities (Issues 3–5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class LayerSpec:
    """One selectable training unit (encoder block, decoder block, or head)."""

    name: str
    kind: str  # encoder | decoder | head | embedding | other
    module_path: str
    index: Optional[int] = None

    def resolve(self, model: nn.Module) -> nn.Module:
        module: nn.Module = model
        for part in self.module_path.split("."):
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        return module


def _param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def get_layer_registry(model: nn.Module) -> Dict[str, LayerSpec]:
    """
    Build a name -> LayerSpec map for NeSymReS Model.

    Encoder blocks:
      encoder_0       = enc.selfatt1          (first ISAB)
      encoder_1..N    = enc.selfatt.{0..}     (remaining ISABs)
      encoder_pma     = enc.outatt            (PMA pooling)

    Decoder blocks:
      decoder_0..M    = decoder_transfomer.layers.{i}

    Heads / embeddings:
      output_head     = fc_out
      tok_embedding   = tok_embedding
      pos_embedding   = pos_embedding
    """
    registry: Dict[str, LayerSpec] = {}

    if not hasattr(model, "enc"):
        raise AttributeError("Expected NeSymReS Model with attribute `enc`")

    enc = model.enc
    # First ISAB
    registry["encoder_0"] = LayerSpec(
        name="encoder_0",
        kind="encoder",
        module_path="enc.selfatt1",
        index=0,
    )
    # Remaining ISABs
    if hasattr(enc, "selfatt"):
        for i, _ in enumerate(enc.selfatt):
            name = f"encoder_{i + 1}"
            registry[name] = LayerSpec(
                name=name,
                kind="encoder",
                module_path=f"enc.selfatt.{i}",
                index=i + 1,
            )
    if hasattr(enc, "outatt"):
        registry["encoder_pma"] = LayerSpec(
            name="encoder_pma",
            kind="encoder",
            module_path="enc.outatt",
            index=None,
        )

    if hasattr(model, "decoder_transfomer"):
        for i, _ in enumerate(model.decoder_transfomer.layers):
            name = f"decoder_{i}"
            registry[name] = LayerSpec(
                name=name,
                kind="decoder",
                module_path=f"decoder_transfomer.layers.{i}",
                index=i,
            )

    if hasattr(model, "fc_out"):
        registry["output_head"] = LayerSpec(
            name="output_head",
            kind="head",
            module_path="fc_out",
        )
    if hasattr(model, "tok_embedding"):
        registry["tok_embedding"] = LayerSpec(
            name="tok_embedding",
            kind="embedding",
            module_path="tok_embedding",
        )
    if hasattr(model, "pos_embedding"):
        registry["pos_embedding"] = LayerSpec(
            name="pos_embedding",
            kind="embedding",
            module_path="pos_embedding",
        )

    return registry


def list_layers(model: nn.Module) -> List[Mapping[str, object]]:
    """Return a tabular summary of selectable layers."""
    registry = get_layer_registry(model)
    rows: List[Mapping[str, object]] = []
    for name, spec in registry.items():
        module = spec.resolve(model)
        n_params = _param_count(module)
        rows.append(
            {
                "name": name,
                "kind": spec.kind,
                "module_path": spec.module_path,
                "index": spec.index,
                "n_params": n_params,
            }
        )
    return rows


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def set_trainable_layers(
    model: nn.Module,
    layer_names: Sequence[str],
    *,
    freeze_others: bool = True,
) -> List[str]:
    """
    Enable gradients only for the named layers.

    Returns the list of activated layer names (resolved).
    """
    registry = get_layer_registry(model)
    unknown = [n for n in layer_names if n not in registry]
    if unknown:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown layer(s): {unknown}. Known: {known}")

    if freeze_others:
        freeze_all(model)

    activated: List[str] = []
    for name in layer_names:
        module = registry[name].resolve(model)
        for p in module.parameters():
            p.requires_grad = True
        activated.append(name)
    return activated


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def trainable_parameter_report(
    model: nn.Module, layer_names: Optional[Iterable[str]] = None
) -> Dict[str, object]:
    """Summarize trainable params, optionally after selecting layers."""
    if layer_names is not None:
        set_trainable_layers(model, list(layer_names), freeze_others=True)
    trainable, total = count_trainable_parameters(model)
    return {
        "trainable": trainable,
        "total": total,
        "fraction": (trainable / total) if total else 0.0,
        "active_layers": list(layer_names) if layer_names is not None else "current",
    }


def assert_only_layers_trainable(
    model: nn.Module, layer_names: Sequence[str]
) -> None:
    """Raise if any trainable param lies outside the selected layers."""
    registry = get_layer_registry(model)
    allowed_ids = set()
    for name in layer_names:
        module = registry[name].resolve(model)
        for p in module.parameters():
            allowed_ids.add(id(p))

    leaks = []
    for pname, p in model.named_parameters():
        if p.requires_grad and id(p) not in allowed_ids:
            leaks.append(pname)
    if leaks:
        raise AssertionError(
            "Trainable parameters outside selected layers:\n  " + "\n  ".join(leaks[:20])
        )
