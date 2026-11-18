"""Fast sanity checks (CPU, < 1 min):  python -m pytest -q tests     or     python tests/test_smoke.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import ALL_MODELS, build_model  # noqa: E402
from segmentation_models.models.lite_med_seg import MicroDeformableAttention  # noqa: E402
from segmentation_models.models.lite_med_seg_ablation import VARIANTS, create_ablation_model  # noqa: E402


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_litemedseg_size_and_output_shape():
    net = build_model("LiteMedSeg", base_channels=12).eval()
    assert n_params(net) == 643_268                       # value reported in the paper (643K)
    with torch.no_grad():
        y = net(torch.randn(2, 3, 256, 256))
    y = y[0] if isinstance(y, (tuple, list)) else y
    assert y.shape == (2, 1, 256, 256)


def test_training_mode_returns_auxiliary_outputs():
    net = build_model("LiteMedSeg", base_channels=12).train()
    out = net(torch.randn(2, 3, 256, 256))
    assert isinstance(out, (tuple, list)) and len(out) == 3   # main + two auxiliary maps (Section 3.7)


def test_hga_gate_receives_gradient():
    """Regression test for the zero-initialisation bug: the descriptor branch must get a non-zero gradient."""
    m = MicroDeformableAttention(32, num_heads=4).train()
    m(torch.randn(2, 32, 8, 8)).sum().backward()
    g = m.descriptor_conv[-1].weight.grad
    assert g is not None and g.abs().sum() > 0
    assert m.descriptor_conv[-1].weight.abs().sum() > 0   # not zero-initialised


def test_all_baselines_forward():
    for name in ALL_MODELS:
        if name == "TransUNet":                            # not part of the final protocol
            continue
        net = build_model(name, base_channels=12).eval()
        with torch.no_grad():
            y = net(torch.randn(1, 3, 256, 256))
        y = y[0] if isinstance(y, (tuple, list)) else y
        assert y.shape[-2:] == (256, 256), name


def test_ablation_variants_build_and_match_full():
    full = n_params(build_model("LiteMedSeg", base_channels=12))
    assert n_params(create_ablation_model("Abl-full_b12")) == full
    for v in VARIANTS:
        net = create_ablation_model(f"Abl-{v}_b12").eval()
        with torch.no_grad():
            assert net(torch.randn(1, 3, 64, 64)).shape[-2:] == (64, 64), v


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
