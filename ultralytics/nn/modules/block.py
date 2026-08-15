# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

_FAST_MODE_CACHE: bool | None = None


def is_fast_mode() -> bool:
    """Check if YOLOv14 Fast Mode is enabled (§4.4).

    When ``YOLOV14_FAST=1``, the scene classifier and DomainAdaptiveLayers
    are skipped at inference, reducing latency while retaining 70% of
    cross-domain gains.

    The result is cached at first call to avoid repeated ``os.environ``
    lookups in hot forward paths.  Call :func:`reset_fast_mode_cache` to
    re-read the environment variable (e.g. in tests).
    """
    global _FAST_MODE_CACHE
    if _FAST_MODE_CACHE is None:
        _FAST_MODE_CACHE = os.environ.get("YOLOV14_FAST", "0") == "1"
    return _FAST_MODE_CACHE


def reset_fast_mode_cache() -> None:
    """Clear the fast-mode cache so the next call re-reads the env var."""
    global _FAST_MODE_CACHE
    _FAST_MODE_CACHE = None


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) for adversarial training — Section 3.3.

    Identity in the forward pass; negates and scales the gradient by
    ``lambd`` in the backward pass.  Placed between the feature extractor
    and the domain classifier so that the feature extractor receives
    reversed gradients (learning domain-invariant features) while the
    classifier receives normal gradients (learning to classify domains).

    The schedule ``lambd = 2/(1+exp(-10·p)) - 1`` (DANN, Ganin et al. 2016)
    gradually increases from 0 to 1 over training, stabilising the
    adversarial game.
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grl_lambda_schedule(epoch: int, max_epochs: int = 100) -> float:
    """DANN-style GRL lambda schedule: 0 → 1 over training.

    ``lambd = 2 / (1 + exp(-10 * p)) - 1`` where ``p = epoch / max_epochs``.
    """
    p = min(epoch / max(max_epochs, 1), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "A2C2f",
    "DeformableAAttn",
    "DeformableABlock",
    "DeformableA2C2f",
    "ViewEmbedding",
    "DynamicScaleRouter",
    "SphereAAttn",
    "DomainAdaptiveLayer",
    "ARMRouter",
    "OffsetSTN",
    "is_fast_mode",
    "reset_fast_mode_cache",
    "GradReverse",
    "grl_lambda_schedule",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        c1 (int): Input channels.
        c2 (): Output channels.
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y

import logging
logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


class DeformableAAttn(nn.Module):
    """
    Deformable Area-Attention with Shifted Windows (D-AAttn) — Section 3.2.

    Extends standard AAttn with:
      1. A learnable deformation field Δ = f_θ(X) that warps the feature grid
         before attention, adapting to geometric distortions.
      2. Local windowed attention (window size W_A × W_A) for linear complexity.
      3. Shifted-window partitioning (shift = ⌊W_A/2⌋) alternated across
         consecutive layers for cross-window information flow (Swin-style),
         with proper attention masks to prevent cross-window leakage.

    Hyper-parameters (paper §3.2):
      W_A = 8 (window size), shift = ⌊W_A/2⌋ = 4.
    """

    def __init__(self, dim, num_heads, area=1, window_size=8, shift_size=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads
        self.window_size = window_size
        self.shift_size = shift_size if shift_size is not None else window_size // 2
        self.dim = dim
        self.all_head_dim = all_head_dim

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

        self.offset_conv = nn.Sequential(
            nn.Conv2d(dim, max(8, dim // 8), 3, 1, 1, bias=False),
            nn.BatchNorm2d(max(8, dim // 8)),
            nn.SiLU(),
            nn.Conv2d(max(8, dim // 8), 2, 3, 1, 1, bias=True),
        )
        nn.init.constant_(self.offset_conv[-1].weight, 0.0)
        nn.init.constant_(self.offset_conv[-1].bias, 0.0)

        self._attn_mask_cache: dict = {}

    def _compute_attn_mask(self, H_p: int, W_p: int, device: torch.device) -> torch.Tensor:
        """Compute Swin-style attention mask for shifted window partitioning.

        After cyclic shift, tokens from different original windows may end up
        in the same shifted window. This mask (value -100 where tokens are
        from different original windows, 0 otherwise) prevents cross-window
        attention leakage when added to the attention logits before softmax.

        Returns
        -------
        torch.Tensor
            Mask of shape (n_win, ws*ws, ws*ws).
        """
        ws = self.window_size
        shift = self.shift_size
        nH, nW = H_p // ws, W_p // ws

        img_mask = torch.zeros((1, H_p, W_p, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = img_mask.reshape(1, nH, ws, nW, ws).permute(0, 1, 3, 2, 4).reshape(nH * nW, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def _get_attn_mask(self, H_p: int, W_p: int, device: torch.device):
        """Get cached attention mask or compute and cache it."""
        key = (H_p, W_p, str(device))
        if key not in self._attn_mask_cache:
            self._attn_mask_cache[key] = self._compute_attn_mask(H_p, W_p, device)
        return self._attn_mask_cache[key]

    @staticmethod
    def _pad_to_window(x: torch.Tensor, ws: int):
        """Pad feature map to be divisible by window size (reflect mode)."""
        H, W = x.shape[2], x.shape[3]
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    @staticmethod
    def _window_partition(x: torch.Tensor, ws: int):
        """Partition padded feature map into windows.

        Args:
            x: (B, C, H_p, W_p) where H_p, W_p are divisible by ws.
        Returns:
            windows: (B*nH*nW, C, ws*ws).
            nH, nW: number of windows in H, W directions.
        """
        B, C, H_p, W_p = x.shape
        nH, nW = H_p // ws, W_p // ws
        windows = x.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, ws * ws)
        return windows, nH, nW

    @staticmethod
    def _window_unpartition(windows: torch.Tensor, B: int, C: int, nH: int, nW: int, ws: int):
        """Re-assemble windows back to feature map."""
        H_p, W_p = nH * ws, nW * ws
        x = windows.reshape(B, nH, nW, C, ws, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H_p, W_p)
        return x

    def forward(self, x):
        """Apply deformable windowed area-attention with shifted windows.

        Pipeline: offset → grid_sample → pad → cyclic_shift → partition →
                  masked_attention → unpartition → reverse_shift → unpad.
        """
        B, C, H, W = x.shape
        device = x.device
        ws = self.window_size
        shift = self.shift_size

        # --- Deformable sampling (skipped in Fast Mode, §4.4) ---
        fast = is_fast_mode()
        if not fast:
            offset = self.offset_conv(x)
            y_g, x_g = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )
            sample_y = y_g.unsqueeze(0) + offset[:, 0, :, :]
            sample_x = x_g.unsqueeze(0) + offset[:, 1, :, :]
            norm_y = 2.0 * sample_y / max(H - 1, 1) - 1.0
            norm_x = 2.0 * sample_x / max(W - 1, 1) - 1.0
            grid = torch.stack((norm_x, norm_y), dim=-1)

        qk = self.qk(x)
        v = self.v(x)

        if not fast:
            qk = F.grid_sample(qk, grid, mode="bilinear", padding_mode="border", align_corners=True)
            v = F.grid_sample(v, grid, mode="bilinear", padding_mode="border", align_corners=True)

        pp = self.pe(v)

        # --- Pad to window size (before shift, §3.2 Swin order) ---
        qk, pad_h, pad_w = self._pad_to_window(qk, ws)
        v, _, _ = self._pad_to_window(v, ws)
        H_p, W_p = qk.shape[2], qk.shape[3]

        if H_p <= ws or W_p <= ws:
            shift = 0

        # --- Cyclic shift ---
        if shift > 0:
            qk = torch.roll(qk, shifts=(-shift, -shift), dims=(2, 3))
            v = torch.roll(v, shifts=(-shift, -shift), dims=(2, 3))

        # --- Window partition ---
        qk_win, nH, nW = self._window_partition(qk, ws)
        v_win, _, _ = self._window_partition(v, ws)
        n_win = qk_win.shape[0]
        n_tokens = ws * ws

        qk_win = qk_win.transpose(1, 2)
        v_win = v_win.transpose(1, 2)

        # --- Attention mask for shifted windows ---
        attn_mask = self._get_attn_mask(H_p, W_p, device) if shift > 0 else None

        # --- Area attention within windows ---
        if self.area > 1:
            qk_win = qk_win.reshape(n_win * self.area, n_tokens // self.area, C * 2)
            v_win = v_win.reshape(n_win * self.area, n_tokens // self.area, C)
            B_a, N_a, _ = qk_win.shape
        else:
            B_a, N_a = n_win, n_tokens

        q, k = qk_win.split([C, C], dim=2)
        q = q.transpose(1, 2).reshape(B_a, self.num_heads, self.head_dim, N_a)
        k = k.transpose(1, 2).reshape(B_a, self.num_heads, self.head_dim, N_a)
        v_win = v_win.transpose(1, 2).reshape(B_a, self.num_heads, self.head_dim, N_a)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)

        if attn_mask is not None and self.area == 1:
            attn = attn.reshape(B, n_win // B, self.num_heads, N_a, N_a)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.reshape(B_a, self.num_heads, N_a, N_a)

        attn = torch.softmax(attn, dim=-1)
        x_out = v_win @ attn.transpose(-2, -1)
        x_out = x_out.permute(0, 3, 1, 2).reshape(B_a, N_a, C)

        if self.area > 1:
            x_out = x_out.reshape(n_win, n_tokens, C)

        # --- Reassemble ---
        x_out = x_out.transpose(1, 2)
        x_out = self._window_unpartition(x_out, B, C, nH, nW, ws)

        if shift > 0:
            x_out = torch.roll(x_out, shifts=(shift, shift), dims=(2, 3))

        if pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :, :H, :W]

        return self.proj(x_out + pp)


class DeformableABlock(nn.Module):
    """Area-Attention block with deformable sampling + MLP.

    Consecutive blocks alternate between regular (shift=0) and shifted
    (shift=⌊W_A/2⌋) window partitioning for cross-window interaction (§3.2).
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1, window_size=8, shift_size=0):
        super().__init__()
        self.attn = DeformableAAttn(dim, num_heads=num_heads, area=area, window_size=window_size, shift_size=shift_size)
        mlp_hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hid, 1), Conv(mlp_hid, dim, 1, act=False))

    def forward(self, x):
        return x + self.mlp(x + self.attn(x))


class DeformableA2C2f(nn.Module):
    """R-ELAN with Deformable Area-Attention blocks.

    Each stage contains paired D-AAttn blocks: the first uses regular window
    partitioning (shift=0), the second uses shifted windows (shift=⌊W_A/2⌋),
    following the Swin Transformer alternating pattern (§3.2).
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True,
                 window_size=8):
        super().__init__()
        c_ = int(c2 * e)
        if c_ % 32 != 0:
            raise ValueError(f"Intermediate channels c_={c_} must be divisible by 32 (got c2={c2}, e={e})")
        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if residual else None
        shift = window_size // 2
        self.m = nn.ModuleList(
            nn.Sequential(
                DeformableABlock(c_, num_heads, mlp_ratio, area, window_size, shift_size=0),  # regular
                DeformableABlock(c_, num_heads, mlp_ratio, area, window_size, shift_size=shift),  # shifted
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return x + self.gamma.view(1, -1, 1, 1) * out if self.gamma is not None else out


class ViewEmbedding(nn.Module):
    """Multi-View Conditioning: injects view-type embedding into feature maps.

    In Fast Mode (§4.4), the embedding injection is skipped and the input
    is returned unchanged to reduce inference latency.
    """

    VIEW_TYPES = {"pinhole": 0, "fisheye": 1, "panoramic": 2, "drone": 3, "bev": 4, "ground": 5}

    def __init__(self, c1, num_views=6, embed_dim=64):
        super().__init__()
        self.view_embed = nn.Embedding(num_views, embed_dim)
        self.proj = Conv(c1 + embed_dim, c1, 1)

    def forward(self, x, view_ids=None):
        if is_fast_mode():
            return x
        B, C, H, W = x.shape
        if view_ids is None:
            view_ids = x.new_zeros(B, dtype=torch.long)
        emb = self.view_embed(view_ids).view(B, -1, 1, 1).expand(-1, -1, H, W)
        return self.proj(torch.cat([x, emb], dim=1))


class DynamicScaleRouter(nn.Module):
    """Adaptive Resolution Pyramid: learns per-input scale importance.

    Generates softmax weights for K pyramid scales.  Optionally fuses
    features using the learned weights.

    Parameters
    ----------
    feat_channels : tuple
        Channel sizes of the K pyramid features.
    reduction : int
        MLP channel reduction ratio.
    return_fused : bool
        If True, also return the weighted-sum fused feature.
    """

    def __init__(self, feat_channels=(256, 512, 1024), reduction=16, return_fused=False):
        super().__init__()
        self.return_fused = return_fused
        self.gate = nn.Sequential(
            nn.Linear(sum(feat_channels), max(16, sum(feat_channels) // reduction)),
            nn.SiLU(),
            nn.Linear(max(16, sum(feat_channels) // reduction), len(feat_channels)),
        )

    def forward(self, features):
        pooled = []
        for f in features:
            if f.dim() == 4:
                pooled.append(F.adaptive_avg_pool2d(f, 1).flatten(1))
            elif f.dim() == 2:
                pooled.append(f)
            else:
                pooled.append(f.flatten(1))
        ctx = torch.cat(pooled, dim=1)
        weights = F.softmax(self.gate(ctx), dim=-1)
        if self.return_fused:
            out = sum(w.view(-1, 1, 1, 1) * f for w, f in zip(weights.unbind(1), features))
            return out, weights
        return weights


class SphereAAttn(nn.Module):
    """Spherical Area-Attention for 360° panoramic images — Section 3.6.

    Partitions the feature map into ``lat_bands`` latitude bands and computes
    self-attention independently within each band.  Following the paper,
    equatorial bands receive **8 heads** while polar bands receive **4 heads**,
    motivated by the cos(φ) information-density scaling.

    Parameters
    ----------
    dim : int           — channel dimension (must be divisible by both
                          ``polar_heads`` and ``equator_heads``).
    num_heads : int     — base number of heads (used for QKV projection).
    lat_bands : int     — number of latitude bands (default 4).
    polar_heads : int   — heads for polar bands (default 4).
    equator_heads : int — heads for equatorial bands (default 8).
    """

    def __init__(self, dim, num_heads=None, lat_bands=4, polar_heads=4, equator_heads=8):
        super().__init__()
        if num_heads is None:
            num_heads = equator_heads
        assert dim % polar_heads == 0, f"dim ({dim}) must be divisible by polar_heads ({polar_heads})"
        assert dim % equator_heads == 0, f"dim ({dim}) must be divisible by equator_heads ({equator_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        ahd = self.head_dim * self.num_heads
        self.all_head_dim = ahd
        self.lat_bands = lat_bands
        self.polar_heads = polar_heads
        self.equator_heads = equator_heads
        self.qk = Conv(dim, ahd * 2, 1, act=False)
        self.v = Conv(dim, ahd, 1, act=False)
        self.proj = Conv(ahd, dim, 1, act=False)
        self.pe = Conv(ahd, dim, 5, 1, 2, g=dim, act=False)

    def _heads_for_band(self, b):
        """Return number of heads for latitude band index *b*.

        Band 0 and band ``lat_bands-1`` are polar (top/bottom of sphere);
        all others are equatorial.
        """
        if b == 0 or b == self.lat_bands - 1:
            return self.polar_heads
        return self.equator_heads

    def _band_bounds(self, H: int):
        """Compute (start, end) row indices for each latitude band.

        Distributes remainder rows across the first few bands so that no
        single band is disproportionately large.
        """
        bs = H // self.lat_bands
        rem = H % self.lat_bands
        bounds = []
        hs = 0
        for b in range(self.lat_bands):
            rows = bs + (1 if b < rem else 0)
            he = hs + rows
            bounds.append((hs, he))
            hs = he
        return bounds

    def forward(self, x):
        B, C, H, W = x.shape
        ahd = self.all_head_dim
        qk = self.qk(x).flatten(2).transpose(1, 2)  # (B, N, 2*ahd)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)  # (B, N, ahd)

        bounds = self._band_bounds(H)
        outs = []
        for b, (hs, he) in enumerate(bounds):
            n_band = (he - hs) * W
            qk_b = qk[:, hs * W:he * W, :]
            v_b = v[:, hs * W:he * W, :]
            q_b, k_b = qk_b.split([ahd, ahd], dim=2)
            nh = self._heads_for_band(b)
            hd = ahd // nh
            q_b = q_b.transpose(1, 2).reshape(B, nh, hd, n_band)
            k_b = k_b.transpose(1, 2).reshape(B, nh, hd, n_band)
            v_b = v_b.transpose(1, 2).reshape(B, nh, hd, n_band)
            attn = torch.softmax((q_b.transpose(-2, -1) @ k_b) * (hd ** -0.5), dim=-1)
            out_b = v_b @ attn.transpose(-2, -1)
            out_b = out_b.permute(0, 3, 1, 2).reshape(B, n_band, ahd)
            outs.append(out_b)
        x_out = torch.cat(outs, dim=1)  # (B, N, ahd)
        x_out = x_out.reshape(B, H, W, ahd).permute(0, 3, 1, 2)
        return self.proj(x_out + pp)


class DomainAdaptiveLayer(nn.Module):
    """
    Game-to-Real Domain Adaptation Layer.

    Uses Adaptive Instance Normalization (AdaIN) to align feature distributions
    between game-rendered and real-photographic domains. This allows the model
    to extract domain-invariant features — a game character and a real person
    produce similar feature representations.

    How it works:
    1. A lightweight domain predictor estimates game-vs-real from features
    2. AdaIN shifts the feature statistics (mean/std) of game features
       toward the real-domain distribution
    3. A domain confusion loss (adversarial) further aligns distributions

    This is placed in the backbone after early conv layers, so downstream
    A2C2f/Detect heads see domain-aligned features.

    Attributes:
        c (int): Channel dimension.
    """

    def __init__(self, c1, c2=None, grl_lambda=0.0):
        """Initialize DomainAdaptiveLayer with AdaIN, domain predictor, and GRL.

        Parameters
        ----------
        c1 : int              — input channels.
        c2 : int, optional    — output channels (default: c1).
        grl_lambda : float    — initial GRL strength (0 = no reversal,
                                annealed to 1 via :func:`grl_lambda_schedule`).
        """
        super().__init__()
        c = c1 if c2 is None else c2

        # Domain classifier (predicts game vs real from feature stats)
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, max(16, c // 16)),
            nn.ReLU(),
            nn.Linear(max(16, c // 16), 2),  # 0=real, 1=game
        )

        # Learned affine transform for AdaIN (game→real alignment)
        self.gamma = nn.Parameter(torch.ones(c))
        self.beta = nn.Parameter(torch.zeros(c))
        self.grl_lambda = grl_lambda

    @staticmethod
    def _instance_norm(x):
        """Compute per-instance mean and std."""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        mean = x_flat.mean(dim=2, keepdim=True).view(B, C, 1, 1)
        std = x_flat.std(dim=2, keepdim=True).view(B, C, 1, 1) + 1e-8
        return mean, std

    @staticmethod
    def _adain(content_feat, gamma, beta, eps=1e-8):
        """Adaptive Instance Normalisation with learned affine params."""
        B, C, H, W = content_feat.shape
        content_flat = content_feat.view(B, C, -1)
        mean = content_flat.mean(dim=2, keepdim=True).view(B, C, 1, 1)
        std = content_flat.std(dim=2, keepdim=True).view(B, C, 1, 1) + eps
        normalized = (content_feat - mean) / std
        return normalized * gamma.view(1, C, 1, 1) + beta.view(1, C, 1, 1)

    def forward(self, x):
        """
        Forward pass with game→real domain adaptation.

        Predicts domain from features, applies AdaIN for game samples,
        stores domain logits for adversarial loss computation.

        In Fast Mode (YOLOV14_FAST=1, §4.4), skips adaptation entirely
        and returns the input unchanged to reduce inference latency.
        """
        if is_fast_mode():
            return x

        B, C, H, W = x.shape

        # Predict domain from feature statistics (with GRL for adversarial training)
        x_reversed = GradReverse.apply(x, self.grl_lambda)
        domain_logits = self.domain_classifier(x_reversed)
        domain_probs = F.softmax(domain_logits, dim=1)
        game_conf = domain_probs[:, 1]

        # Apply AdaIN adaptation with smooth blending
        x_adapted = self._adain(x, self.gamma, self.beta)
        mask = game_conf.view(B, 1, 1, 1).expand(-1, C, H, W).sigmoid()
        x_out = (1 - mask) * x + mask * x_adapted

        # Store for external loss access (per-batch; overwritten each forward)
        self._domain_logits = domain_logits
        self._domain_probs = domain_probs

        return x_out

    def get_domain_logits(self):
        """Return the most recent batch's domain logits (for loss computation)."""
        return getattr(self, "_domain_logits", None)


class ARMRouter(nn.Module):
    """Adaptive Routing and Modulation (ARM) framework — Section 3.1.

    Unifies all adaptive components under:
        y = F_base(x) + Σ_k γ_k(x) · T_k(x)
    where γ_k = σ(MLP_k(Pool(x))) are input-dependent routing weights.

    When ``transforms`` are provided, the full ARM formula is computed.
    When ``transforms`` is None, only the routing weights γ are returned
    (backward-compatible with external composition).

    Parameters
    ----------
    c : int              — feature channel dimension.
    n_transforms : int   — number of specialised transforms K.
    reduction : int      — MLP channel reduction ratio.
    transforms : list, optional
        ModuleList of K transform modules T_k.  If None, default
        transforms are created: identity, 1×1 conv, 3×3 dwconv, 1×1 conv.
    base_module : nn.Module, optional
        Base transform F_base.  If None, identity is used.
    """

    def __init__(self, c, n_transforms=4, reduction=16, transforms=None, base_module=None):
        super().__init__()
        self.n_transforms = n_transforms
        hidden = max(16, c // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_transforms),
        )
        self.base = base_module if base_module is not None else nn.Identity()
        if transforms is not None:
            self.transforms = nn.ModuleList(transforms)
        else:
            self.transforms = nn.ModuleList([
                nn.Identity(),
                Conv(c, c, 1, act=False),
                DWConv(c, c, 3),
                Conv(c, c, 1, act=False),
            ])

    def forward(self, x):
        """Compute y = F_base(x) + Σ_k γ_k(x) · T_k(x).

        If the module has no transforms (empty list), returns just γ weights
        for backward compatibility with external composition.
        """
        gamma = torch.sigmoid(self.gate(x))  # (B, K)
        if len(self.transforms) == 0:
            return gamma
        y = self.base(x)
        for k, T_k in enumerate(self.transforms):
            y = y + gamma[:, k].view(-1, 1, 1, 1) * T_k(x)
        return y


class OffsetSTN(nn.Module):
    """Lightweight Spatial Transformer Network for offset regularisation target.

    Pre-trained on synthetic fisheye pairs to estimate physically plausible
    deformation fields.  Used **only during training** to generate Δ_init;
    at inference (eval mode) it returns a zero offset field, avoiding
    unnecessary computation (Section 3.2).
    """

    def __init__(self, c):
        super().__init__()
        self.stn = nn.Sequential(
            nn.Conv2d(c, max(8, c // 8), 3, 1, 1, bias=False),
            nn.BatchNorm2d(max(8, c // 8)),
            nn.SiLU(),
            nn.Conv2d(max(8, c // 8), 2, 3, 1, 1, bias=True),
        )
        nn.init.constant_(self.stn[-1].weight, 0.0)
        nn.init.constant_(self.stn[-1].bias, 0.0)

    def forward(self, x):
        """Return target offset field Δ_init ∈ R^{B×2×H×W}.

        During inference (``self.training == False``), returns zeros to
        skip computation — the STN is only needed for training loss.
        """
        if not self.training:
            return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        return self.stn(x)