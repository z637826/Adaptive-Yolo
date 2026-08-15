"""
YOLOv14 Dummy Data Loader.

Synthetic data generator that produces fake images and annotations for
training/validation without touching real datasets.  This lets the full
YOLOv14 pipeline (forward, loss, backward) run end-to-end for smoke tests,
unit tests, and CI — exactly as required by the NeurIPS 2024 paper but
without any external data dependency.

The loader generates:
  * Random RGB images (torch.randn) of configurable size.
  * Random bounding boxes with class labels.
  * Per-image view-type labels  ∈ {pinhole, fisheye, panoramic, drone, BEV, ground}
    for Multi-View Conditioning (Section 3.4).
  * Per-image domain labels    ∈ {real, game}
    for Game2Real Domain Adaptation (Section 3.3).
  * Class-balanced sampling: each batch contains 3 samples per class across
    2-3 different views (Section 3.4, Cross-View Contrastive Loss).
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Tuple

import torch
from torch.utils.data import Dataset

# Six view types defined in the paper (Section 3.4)
VIEW_TYPES: Dict[str, int] = {
    "pinhole": 0,
    "fisheye": 1,
    "panoramic": 2,
    "drone": 3,
    "bev": 4,
    "ground": 5,
}
VIEW_NAMES: List[str] = list(VIEW_TYPES.keys())

# Two domains for Game2Real adaptation (Section 3.3)
DOMAIN_REAL = 0
DOMAIN_GAME = 1

# Class-balanced sampling parameters (Section 3.4)
SAMPLES_PER_CLASS = 3  # each batch contains 3 samples per class
RARE_CLASS_THRESHOLD = 10  # classes with <10 images: include all available
NUM_VIEWS_PER_CLASS = (2, 3)  # spread across 2-3 different views


class DummyDetectionDataset(Dataset):
    """Synthetic detection dataset that generates random images + boxes.

    Parameters
    ----------
    n_samples : int
        Number of synthetic samples in the dataset.
    img_size : int
        Square image resolution (default 640, matching paper Section 4.1).
    nc : int
        Number of object classes (default 80, COCO).
    max_boxes : int
        Maximum number of boxes per image.
    channels : int
        Image channels (3 for RGB).
    """

    def __init__(
        self,
        n_samples: int = 100,
        img_size: int = 640,
        nc: int = 80,
        max_boxes: int = 10,
        channels: int = 3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.img_size = img_size
        self.nc = nc
        self.max_boxes = max_boxes
        self.channels = channels
        # Pre-generate per-sample metadata so __getitem__ is deterministic
        self._meta = [self._sample_meta(i) for i in range(n_samples)]
        # Build class → [sample_indices] index for class-balanced sampling
        self._class_index: Dict[int, List[int]] = {c: [] for c in range(nc)}
        for i, meta in enumerate(self._meta):
            for box in meta["boxes"]:
                cls_id = int(box[0])
                if i not in self._class_index[cls_id]:
                    self._class_index[cls_id].append(i)

    def _sample_meta(self, idx: int) -> Dict:
        rng = random.Random(idx)
        n_boxes = rng.randint(1, self.max_boxes)
        boxes = []
        for _ in range(n_boxes):
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            w = rng.uniform(0.05, 0.4)
            h = rng.uniform(0.05, 0.4)
            cls = rng.randint(0, self.nc - 1)
            boxes.append((cls, cx, cy, w, h))
        return {
            "boxes": boxes,
            "view_id": rng.randint(0, len(VIEW_TYPES) - 1),
            "domain": rng.choice([DOMAIN_REAL, DOMAIN_GAME]),
        }

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self._meta[idx]
        # Random image ~ N(0.5, 0.25) clipped to [0, 1] range
        img = torch.randn(self.channels, self.img_size, self.img_size) * 0.25 + 0.5
        # Build box tensor  (N, 5)  cls + xywh normalised
        boxes = torch.tensor(meta["boxes"], dtype=torch.float32)  # (N, 5)
        if boxes.numel() == 0:
            boxes = torch.zeros(0, 5)
        return {
            "img": img,
            "boxes": boxes,
            "view_id": torch.tensor(meta["view_id"], dtype=torch.long),
            "domain": torch.tensor(meta["domain"], dtype=torch.long),
            "cls": boxes[:, 0].long() if boxes.numel() else torch.zeros(0, dtype=torch.long),
        }

    def get_class_indices(self, cls_id: int) -> List[int]:
        """Return all sample indices containing the given class."""
        return self._class_index.get(cls_id, [])


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate variable-length box lists into padded tensors."""
    imgs = torch.stack([b["img"] for b in batch], dim=0)  # (B, C, H, W)
    view_ids = torch.stack([b["view_id"] for b in batch], dim=0)  # (B,)
    domains = torch.stack([b["domain"] for b in batch], dim=0)  # (B,)
    max_n = max(b["boxes"].shape[0] for b in batch)
    B = len(batch)
    boxes = torch.zeros(B, max_n, 5)
    box_mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["boxes"].shape[0]
        if n > 0:
            boxes[i, :n] = b["boxes"]
            box_mask[i, :n] = True
    return {
        "img": imgs,
        "boxes": boxes,
        "box_mask": box_mask,
        "view_ids": view_ids,
        "domains": domains,
    }


class ClassBalancedSampler:
    """Class-balanced sampler for cross-view contrastive learning (§3.4).

    Each batch contains ``samples_per_class`` samples per class, spread across
    2-3 different view types.  For rare classes (fewer than
    ``rare_class_threshold`` images), all available samples are included.

    Parameters
    ----------
    dataset : DummyDetectionDataset
        Dataset with ``get_class_indices`` method.
    batch_size : int
        Target batch size (paper: 256).  Actual batch may be smaller if
        there are not enough classes/samples to fill it.
    samples_per_class : int
        Number of samples per class per batch (paper: 3).
    rare_class_threshold : int
        Classes with fewer than this many images use all available samples.
    num_views_range : tuple
        (min, max) number of distinct views to spread samples across.
    """

    def __init__(
        self,
        dataset: DummyDetectionDataset,
        batch_size: int = 256,
        samples_per_class: int = SAMPLES_PER_CLASS,
        rare_class_threshold: int = RARE_CLASS_THRESHOLD,
        num_views_range: Tuple[int, int] = NUM_VIEWS_PER_CLASS,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.spc = samples_per_class
        self.rare_threshold = rare_class_threshold
        self.nv_min, self.nv_max = num_views_range
        self.nc = dataset.nc
        # Determine how many classes to sample per batch
        self.classes_per_batch = max(1, batch_size // samples_per_class)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield lists of sample indices forming class-balanced batches."""
        rng = random.Random()
        n_batches = len(self.dataset) // self.batch_size
        for _ in range(n_batches):
            batch_indices: List[int] = []
            # Randomly select classes for this batch
            selected_classes = rng.sample(range(self.nc), min(self.classes_per_batch, self.nc))
            for cls_id in selected_classes:
                candidates = self.dataset.get_class_indices(cls_id)
                if len(candidates) == 0:
                    continue
                # For rare classes, use all available; otherwise sample spc
                if len(candidates) < self.rare_threshold:
                    chosen = candidates[:]
                else:
                    chosen = rng.sample(candidates, min(self.spc, len(candidates)))
                # Try to spread across 2-3 different views
                chosen = self._diversify_views(cls_id, chosen, rng)
                batch_indices.extend(chosen)
            if len(batch_indices) == 0:
                # Fallback: random indices
                batch_indices = rng.sample(range(len(self.dataset)), min(self.batch_size, len(self.dataset)))
            yield batch_indices

    def _diversify_views(self, cls_id: int, candidates: List[int], rng: random.Random) -> List[int]:
        """Select samples from different views for cross-view positives.

        Actively picks samples from 2-3 distinct view types when available,
        rather than just reordering existing samples.  This ensures that
        each batch contains genuine cross-view positive pairs for the
        contrastive loss (Section 3.4).
        """
        by_view: Dict[int, List[int]] = {}
        for idx in candidates:
            vid = self.dataset._meta[idx]["view_id"]
            by_view.setdefault(vid, []).append(idx)

        if len(by_view) <= 1:
            return candidates

        n_views = rng.randint(self.nv_min, min(self.nv_max, len(by_view)))
        selected_views = rng.sample(list(by_view.keys()), n_views)
        result: List[int] = []
        per_view = max(1, self.spc // n_views)
        for vid in selected_views:
            result.extend(rng.sample(by_view[vid], min(len(by_view[vid]), per_view)))
        return result


class DummyDataLoader:
    """Iterable dummy data loader wrapping ``DummyDetectionDataset``.

    Mimics the minimal interface needed by the YOLOv14 training loop:
    ``for batch in loader: ...`` where each batch is a dict produced by
    :func:`collate_fn`.

    When ``class_balanced=True``, uses :class:`ClassBalancedSampler` to
    construct batches with 3 samples per class across 2-3 views (§3.4).
    """

    def __init__(
        self,
        n_samples: int = 100,
        batch_size: int = 4,
        img_size: int = 640,
        nc: int = 80,
        max_boxes: int = 10,
        shuffle: bool = True,
        class_balanced: bool = False,
    ):
        self.dataset = DummyDetectionDataset(n_samples, img_size, nc, max_boxes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_balanced = class_balanced
        self._indices = list(range(n_samples))
        if class_balanced:
            self._sampler = ClassBalancedSampler(self.dataset, batch_size=batch_size)
        else:
            self._sampler = None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.class_balanced and self._sampler is not None:
            for batch_idx in self._sampler:
                batch = [self.dataset[j] for j in batch_idx]
                yield collate_fn(batch)
        else:
            idxs = self._indices[:]
            if self.shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch_idx = idxs[i : i + self.batch_size]
                batch = [self.dataset[j] for j in batch_idx]
                yield collate_fn(batch)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def build_dummy_loader(
    split: str = "train",
    batch_size: int = 4,
    img_size: int = 640,
    nc: int = 80,
    n_samples: int | None = None,
    class_balanced: bool = False,
) -> DummyDataLoader:
    """Convenience factory used as a drop-in for ``build_dataloader``.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"val"`` — only affects default sample count.
    class_balanced : bool
        If True, enable class-balanced sampling for cross-view contrastive
        learning (§3.4).
    """
    if n_samples is None:
        n_samples = 256 if split == "train" else 64
    return DummyDataLoader(
        n_samples=n_samples,
        batch_size=batch_size,
        img_size=img_size,
        nc=nc,
        shuffle=(split == "train"),
        class_balanced=class_balanced,
    )