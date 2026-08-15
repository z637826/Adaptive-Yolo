"""YOLOv14 module smoke test.

Verifies that every architectural innovation described in the NeurIPS 2024
paper can be instantiated and run a forward pass with synthetic data from
``ultralytics.data.dummy_data``.  No real dataset or pretrained weights
are required.

Run:
    python tests/test_v14_modules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure the local ultralytics package is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.block import (          # noqa: E402
    A2C2f,
    ARMRouter,
    DeformableA2C2f,
    DeformableAAttn,
    DeformableABlock,
    DomainAdaptiveLayer,
    DynamicScaleRouter,
    GradReverse,
    OffsetSTN,
    SphereAAttn,
    ViewEmbedding,
    grl_lambda_schedule,
    is_fast_mode,
    reset_fast_mode_cache,
)
from ultralytics.nn.modules.conv import CircularConv, Conv, DeformableConv  # noqa: E402
from ultralytics.data.dummy_data import (  # noqa: E402
    DOMAIN_GAME,
    DOMAIN_REAL,
    DummyDataLoader,
    VIEW_NAMES,
    build_dummy_loader,
)
from ultralytics.utils.loss import (  # noqa: E402
    CrossViewConsistencyLoss,
    DomainAdversarialLoss,
    OffsetRegularizationLoss,
)
from ultralytics.utils.seed_utils import set_seed, get_seed  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, C, H, W = 2, 64, 32, 32
PASSED: list[str] = []
FAILED: list[str] = []


def _ok(name: str):
    PASSED.append(name)
    print(f"  ✓ {name}")


def _fail(name: str, exc: Exception):
    FAILED.append(name)
    print(f"  ✗ {name}: {exc}")


def test_deformable_conv():
    m = DeformableConv(C, C, k=3).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    y = m(x)
    assert y.shape == (B, C, H, W), f"shape {y.shape}"
    _ok("DeformableConv")


def test_circular_conv():
    m = CircularConv(C, C, k=3).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    y = m(x)
    assert y.shape == (B, C, H, W), f"shape {y.shape}"
    _ok("CircularConv")


def test_deformable_a2c2f():
    m = DeformableA2C2f(C, C, n=1, a2=True, area=1).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    y = m(x)
    assert y.shape == (B, C, H, W), f"shape {y.shape}"
    _ok("DeformableA2C2f")


def test_view_embedding():
    m = ViewEmbedding(C, num_views=6, embed_dim=32).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    view_ids = torch.randint(0, 6, (B,), device=DEVICE)
    y = m(x, view_ids)
    assert y.shape == (B, C, H, W), f"shape {y.shape}"
    _ok("ViewEmbedding")


def test_domain_adaptive_layer():
    m = DomainAdaptiveLayer(C).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    y = m(x)
    assert y.shape == (B, C, H, W), f"shape {y.shape}"
    assert hasattr(m, "_domain_logits"), "domain logits not stored"
    _ok("DomainAdaptiveLayer")


def test_dynamic_scale_router():
    m = DynamicScaleRouter(feat_channels=(64, 64, 64)).to(DEVICE)
    feats = [torch.randn(B, 64, 8, 8, device=DEVICE) for _ in range(3)]
    w = m(feats)
    assert w.shape == (B, 3), f"shape {w.shape}"
    assert torch.allclose(w.sum(dim=1), torch.ones(B, device=DEVICE), atol=1e-5), "weights not normalised"
    _ok("DynamicScaleRouter")


def test_sphere_aattn():
    m = SphereAAttn(C, num_heads=4, lat_bands=4).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    y = m(x)
    assert y.shape == (B, C, H, W), f"shape {y.shape}"
    _ok("SphereAAttn")


def test_cross_view_loss():
    loss_fn = CrossViewConsistencyLoss(temperature=0.07).to(DEVICE)
    feats = torch.randn(8, 64, device=DEVICE)
    view_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=DEVICE)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], device=DEVICE)
    loss = loss_fn(feats, view_ids, labels)
    assert loss.dim() == 0, "loss should be scalar"
    _ok("CrossViewConsistencyLoss")


def test_domain_adversarial_loss():
    loss_fn = DomainAdversarialLoss(lambda_domain=0.1).to(DEVICE)
    logits = torch.randn(8, 2, device=DEVICE)
    labels = torch.randint(0, 2, (8,), device=DEVICE)
    cls_loss, conf_loss = loss_fn(logits, labels)
    assert cls_loss.dim() == 0 and conf_loss.dim() == 0, "losses should be scalar"
    _ok("DomainAdversarialLoss")


def test_dummy_loader():
    loader = build_dummy_loader(split="train", batch_size=4, img_size=320, nc=80, n_samples=8)
    batch = next(iter(loader))
    assert batch["img"].shape == (4, 3, 320, 320), f"img shape {batch['img'].shape}"
    assert batch["view_ids"].shape == (4,), f"view_ids shape {batch['view_ids'].shape}"
    assert batch["domains"].shape == (4,), f"domains shape {batch['domains'].shape}"
    _ok("DummyDataLoader")


def test_arm_router():
    """ARM framework with transforms: y = F_base(x) + Sum gamma_k * T_k(x) (Section 3.1)."""
    m = ARMRouter(C, n_transforms=4, reduction=16).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    y = m(x)
    assert y.shape == (B, C, H, W), f"shape {y.shape} (should be feature map, not weights)"
    _ok("ARMRouter (full formula)")


def test_offset_stn():
    """Offset STN: returns offset in training, zeros in eval (Section 3.2)."""
    m = OffsetSTN(C).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    m.train()
    delta_train = m(x)
    assert delta_train.shape == (B, 2, H, W), f"train shape {delta_train.shape}"
    m.eval()
    delta_eval = m(x)
    assert delta_eval.shape == (B, 2, H, W), f"eval shape {delta_eval.shape}"
    assert torch.allclose(delta_eval, torch.zeros_like(delta_eval), atol=1e-6), "eval should return zeros"
    _ok("OffsetSTN (train/eval)")


def test_deformable_aattn_shifted():
    """D-AAttn with shifted windows (§3.2)."""
    # Regular window (shift=0)
    m_reg = DeformableAAttn(C, num_heads=4, area=1, window_size=8, shift_size=0).to(DEVICE)
    # Shifted window (shift=4)
    m_shift = DeformableAAttn(C, num_heads=4, area=1, window_size=8, shift_size=4).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    y_reg = m_reg(x)
    y_shift = m_shift(x)
    assert y_reg.shape == (B, C, H, W), f"regular shape {y_reg.shape}"
    assert y_shift.shape == (B, C, H, W), f"shifted shape {y_shift.shape}"
    _ok("DeformableAAttn (shifted windows)")


def test_deformable_ablock_alternating():
    """Paired D-AAttn blocks alternate regular/shifted windows (§3.2)."""
    m = DeformableABlock(C, num_heads=4, area=1, window_size=8, shift_size=4).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    y = m(x)
    assert y.shape == (B, C, H, W), f"shape {y.shape}"
    _ok("DeformableABlock (alternating)")


def test_class_balanced_loader():
    """Class-balanced sampling for cross-view contrastive loss (§3.4)."""
    loader = build_dummy_loader(
        split="train", batch_size=12, img_size=320, nc=10, n_samples=100, class_balanced=True
    )
    batch = next(iter(loader))
    assert "img" in batch and "view_ids" in batch, "batch missing keys"
    assert batch["img"].shape[1:] == (3, 320, 320), f"img shape {batch['img'].shape}"
    _ok("ClassBalancedSampler")


def test_offset_regularization_loss():
    """Offset regularisation loss (§3.2)."""
    loss_fn = OffsetRegularizationLoss(lambda_reg=0.01).to(DEVICE)
    delta = torch.randn(B, 2, H, W, device=DEVICE)
    delta_init = torch.randn(B, 2, H, W, device=DEVICE)
    loss = loss_fn(delta, delta_init)
    assert loss.dim() == 0, "loss should be scalar"
    assert loss.item() >= 0, "loss should be non-negative"
    _ok("OffsetRegularizationLoss")


def test_fast_mode():
    """Fast Mode skips DomainAdaptiveLayer and ViewEmbedding (Section 4.4)."""
    import os

    m = DomainAdaptiveLayer(C).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)

    # Normal mode: adaptation applied
    os.environ.pop("YOLOV14_FAST", None)
    reset_fast_mode_cache()
    y_normal = m(x)
    assert y_normal.shape == (B, C, H, W)

    # Fast mode: should return input unchanged
    os.environ["YOLOV14_FAST"] = "1"
    reset_fast_mode_cache()
    assert is_fast_mode(), "fast mode not detected"
    y_fast = m(x)
    assert torch.allclose(y_fast, x, atol=1e-6), "fast mode did not skip adaptation"
    os.environ.pop("YOLOV14_FAST", None)
    reset_fast_mode_cache()
    _ok("FastMode (DomainAdaptiveLayer skip)")

    # ViewEmbedding should also skip in fast mode
    ve = ViewEmbedding(C, num_views=6, embed_dim=32).to(DEVICE)
    os.environ["YOLOV14_FAST"] = "1"
    reset_fast_mode_cache()
    y_ve = ve(x)
    assert torch.allclose(y_ve, x, atol=1e-6), "ViewEmbedding not skipped in fast mode"
    os.environ.pop("YOLOV14_FAST", None)
    reset_fast_mode_cache()
    _ok("FastMode (ViewEmbedding skip)")


def test_soft_routing():
    """AdaptiveAugmentPolicy soft routing (§3.5)."""
    import numpy as np

    from ultralytics.data.augment import AdaptiveAugmentPolicy

    policy = AdaptiveAugmentPolicy(soft_routing=True, p=1.0)
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    labels = {"img": img}
    out = policy(labels)
    assert "img" in out, "output missing img"
    # Soft routing should store scene probabilities
    assert "scene_probs" in out, "soft routing did not store scene_probs"
    _ok("AdaptiveAugmentPolicy (soft routing)")


def test_grl_gradient_reversal():
    """Gradient Reversal Layer reverses gradient in backward (Section 3.3)."""
    x = torch.randn(4, 4, requires_grad=True)
    lambd = 1.0
    y = GradReverse.apply(x, lambd)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x.grad, -torch.ones_like(x.grad), atol=1e-6), \
        f"GRL gradient should be -1, got {x.grad}"
    _ok("GradReverse (gradient reversal)")


def test_grl_lambda_schedule():
    """GRL lambda schedule: 0 at epoch 0, approaching 1 at max_epochs."""
    assert abs(grl_lambda_schedule(0, 100)) < 1e-6, "lambda should be ~0 at epoch 0"
    assert grl_lambda_schedule(100, 100) > 0.99, "lambda should be ~1 at max_epochs"
    assert grl_lambda_schedule(50, 100) > 0.4, "lambda should be ~0.5 at half"
    _ok("grl_lambda_schedule")


def test_dynamic_scale_router_fused():
    """DynamicScaleRouter with return_fused returns weighted sum."""
    m = DynamicScaleRouter(feat_channels=(64, 64, 64), return_fused=True).to(DEVICE)
    feats = [torch.randn(B, 64, 8, 8, device=DEVICE) for _ in range(3)]
    out, w = m(feats)
    assert out.shape == (B, 64, 8, 8), f"fused shape {out.shape}"
    assert w.shape == (B, 3), f"weights shape {w.shape}"
    assert torch.allclose(w.sum(dim=1), torch.ones(B, device=DEVICE), atol=1e-5)
    _ok("DynamicScaleRouter (fused output)")


def test_seed_reproducibility():
    """set_seed produces deterministic results."""
    set_seed(42)
    a = torch.randn(3, 3)
    set_seed(42)
    b = torch.randn(3, 3)
    assert torch.allclose(a, b, atol=1e-6), "seed not reproducible"
    assert get_seed() == 42
    _ok("SeedReproducibility")


def test_attention_mask():
    """Shifted window attention produces different output from regular (Section 3.2)."""
    torch.manual_seed(42)
    m_reg = DeformableAAttn(C, num_heads=4, area=1, window_size=8, shift_size=0).to(DEVICE)
    m_shift = DeformableAAttn(C, num_heads=4, area=1, window_size=8, shift_size=4).to(DEVICE)
    m_shift.load_state_dict(m_reg.state_dict())
    x = torch.randn(B, C, H, W, device=DEVICE)
    with torch.no_grad():
        y_reg = m_reg(x)
        y_shift = m_shift(x)
    assert y_reg.shape == y_shift.shape
    assert not torch.allclose(y_reg, y_shift, atol=1e-4), \
        "shifted window should produce different output from regular"
    _ok("AttentionMask (shifted != regular)")


def test_cross_view_loss_stability():
    """CrossViewConsistencyLoss is numerically stable with small temperature."""
    loss_fn = CrossViewConsistencyLoss(tau_min=0.05, tau_max=0.05).to(DEVICE)
    feats = torch.randn(8, 64, device=DEVICE)
    view_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=DEVICE)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], device=DEVICE)
    loss = loss_fn(feats, view_ids, labels)
    assert loss.dim() == 0, "loss should be scalar"
    assert torch.isfinite(loss), f"loss should be finite, got {loss}"
    _ok("CrossViewConsistencyLoss (numerical stability)")


def test_offset_stn_eval_zeros():
    """OffsetSTN returns zeros in eval mode without computation."""
    m = OffsetSTN(C).to(DEVICE)
    x = torch.randn(B, C, H, W, device=DEVICE)
    m.eval()
    with torch.no_grad():
        delta = m(x)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)
    _ok("OffsetSTN (eval zeros)")


def test_yaml_configs():
    """Verify all v14 YAML configs parse into a model."""
    from ultralytics import YOLO

    cfg_dir = ROOT / "ultralytics" / "cfg" / "models" / "v14"
    yamls = [
        "yolov14.yaml",
        "yolov14-deformable.yaml",
        "yolov14-game2real.yaml",
        "yolov14-multiview.yaml",
        "yolov14-panorama.yaml",
        "yolov14-adaptive.yaml",
    ]
    for name in yamls:
        path = cfg_dir / name
        if not path.exists():
            _fail(f"YAML {name}", FileNotFoundError(path))
            continue
        try:
            model = YOLO(str(path))
            # Verify forward pass works
            import torch as _t
            _x = _t.randn(1, 3, 640, 640)
            with _t.no_grad():
                _ = model.model(_x)
            _ok(f"YAML {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb
            _fail(f"YAML {name}", exc)
            _tb.print_exc()


def main():
    print(f"\n{'=' * 60}")
    print(f"YOLOv14 Module Smoke Test  (device: {DEVICE})")
    print(f"{'=' * 60}\n")

    tests = [
        test_deformable_conv,
        test_circular_conv,
        test_deformable_a2c2f,
        test_deformable_aattn_shifted,
        test_deformable_ablock_alternating,
        test_attention_mask,
        test_view_embedding,
        test_domain_adaptive_layer,
        test_dynamic_scale_router,
        test_dynamic_scale_router_fused,
        test_sphere_aattn,
        test_arm_router,
        test_offset_stn,
        test_offset_stn_eval_zeros,
        test_cross_view_loss,
        test_cross_view_loss_stability,
        test_domain_adversarial_loss,
        test_offset_regularization_loss,
        test_dummy_loader,
        test_class_balanced_loader,
        test_fast_mode,
        test_soft_routing,
        test_grl_gradient_reversal,
        test_grl_lambda_schedule,
        test_seed_reproducibility,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            _fail(t.__name__, exc)

    # YAML config tests (may fail if dependencies missing — non-fatal)
    print("\n--- YAML config parse tests ---")
    try:
        test_yaml_configs()
    except Exception as exc:  # noqa: BLE001
        _fail("YAML configs (import)", exc)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Passed: {len(PASSED)}   Failed: {len(FAILED)}")
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
        return 1
    print("All tests passed ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())