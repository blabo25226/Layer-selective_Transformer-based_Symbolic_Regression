"""Unit tests for layer_selector without loading full checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 12):
    pytest.skip("NeSymReS/Hydra 1.0 requires Python 3.10 or 3.11", allow_module_level=True)
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from nesymres.architectures.model import Model  # noqa: E402
from models.layer_selector import (  # noqa: E402
    assert_only_layers_trainable,
    count_trainable_parameters,
    get_layer_registry,
    list_layers,
    set_trainable_layers,
)


def _tiny_cfg():
    return SimpleNamespace(
        trg_pad_idx=0,
        output_dim=32,
        dim_hidden=32,
        length_eq=16,
        sinuisodal_embeddings=False,
        num_heads=4,
        dec_pf_dim=64,
        dropout=0.0,
        linear=False,
        bit16=True,
        norm=True,
        mean=0.5,
        std=0.5,
        activation="relu",
        input_normalization=False,
        dim_input=4,
        num_inds=8,
        ln=True,
        n_l_enc=2,
        num_features=8,
        dec_layers=2,
    )


def test_registry_and_freeze():
    cfg = _tiny_cfg()
    model = Model(cfg)
    rows = list_layers(model)
    names = {r["name"] for r in rows}

    assert "encoder_0" in names
    assert "encoder_1" in names
    assert "encoder_2" in names
    assert "encoder_pma" in names
    assert "decoder_0" in names
    assert "decoder_1" in names
    assert "output_head" in names

    reg = get_layer_registry(model)
    set_trainable_layers(model, ["decoder_1"], freeze_others=True)
    assert_only_layers_trainable(model, ["decoder_1"])
    trainable, total = count_trainable_parameters(model)
    assert 0 < trainable < total

    expected = sum(p.numel() for p in reg["decoder_1"].resolve(model).parameters())
    assert trainable == expected


if __name__ == "__main__":
    test_registry_and_freeze()
    print("tests passed")
