<h1 align="center">🚀 YOLO：Unified Cross-Domain Real‑Time Object Detection with Adaptive Multi‑View Representation</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2608.04720"><img src="https://img.shields.io/badge/arXiv-2608.04720-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://github.com/yolo"><img src="https://img.shields.io/badge/GitHub-z6378241/yolo-181717?style=flat-square&logo=github" alt="GitHub"></a>
<p align="center">
  <a href="https://cheinralational.github.io/JianLu.io/"><img src="https://img.shields.io/badge/First_Author-Jian_Lu-blue?style=flat-square" alt="First Author"></a>
  <a href="https://baike.baidu.com"><img src="https://img.shields.io/badge/Corresponding_Author-Chenbin_Zhang-green?style=flat-square" alt="Corresponding Author"></a>
</p>
<p align="center">
  <strong>The only real‑time detector that exceeds 43 mAP on all four challenging benchmarks – game, fisheye, drone, and panorama – simultaneously.</strong>
</p>

---

## 📋 Table of Contents

- [Why YOLO?](#-why-yolo)
- [Performance Highlights](#-performance-highlights)
- [Project Roadmap & Status](#-project-roadmap--status)
- [Architecture Overview](#-architecture-overview)
- [Core Components](#-core-components)
- [Model Variants](#-model-variants)
- [Quick Start](#-quick-start)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Why YOLO?

Conventional detectors excel under ideal pinhole‑camera conditions, but degrade sharply in **real‑world non‑ideal imaging** scenarios. YOLOv14 learns **domain‑invariant, viewpoint‑robust** features via a combination of deformable attention, adaptive instance normalisation, and adversarial domain alignment:

| Scenario | Problem | YOLO Solution |
|----------|---------|------------------|
| **Fisheye / wide‑angle** | Barrel distortion shifts and compresses objects near edges | Deformable Area‑Attention (D‑AAttn) warps the feature grid to compensate for distortion |
| **Game footage** (Delta Force, COD, PUBG) | Rendering style (posterisation, edge sharpening, high saturation) causes missed detections | Game2Real domain adaptation with AdaIN + adversarial domain classifier aligns feature distributions |
| **Drone / top‑down view** | Unfamiliar scales and viewpoints, dense small objects | Multi‑view conditioning (ViewEmbedding) adapts to aerial perspectives |
| **360° panoramas** | Latitude stretching and 0°/360° boundary discontinuity | Spherical Attention (SphereAAttn) + CircularConv handle equirectangular projection |

---

## 📊 Performance Highlights

| Metric | Value |
|--------|-------|
| **COCO mAP** (val2017, s‑scale) | **49.1** |
| **Latency** (T4 TensorRT FP16) | **2.91 ms** |
| **Throughput** | **344 FPS** |
| **Game benchmark** | **50.2 mAP** (+26.1 ↑ over YOLOv12s) |
| **Panorama benchmark** | **45.1 mAP** (+6.6 ↑ over best baseline) |
| **Drone benchmark** | **43.2 mAP** (+6.4 ↑ over best baseline) |
| **Fisheye benchmark** | **45.3 mAP** (+4.1 ↑ over best baseline) |

> **YOLO is the only real‑time detector that exceeds 43 mAP on all four challenging benchmarks at the same time.**

---

## 🗺️ Project Roadmap & Status

> **Last updated:** August 2026

| Status | Task | Description |
|--------|------|-------------|
| ✅ **DONE** | **arXiv technical report** | Full paper (2608.04720) released with mathematical derivations, ablation studies, and benchmark comparisons. |
| ✅ **DONE** | **Codebase (architecture & modules)** | Complete training/inference pipeline open‑sourced, including `DeformableAAttn`, `DomainAdaptiveLayer`, `SphereAAttn`, `ViewEmbedding`, `DynamicScaleRouter`, and all YAML configs. |
| ✅ **DONE** | **Local web demo** | `app.py` (Gradio) is ready for immediate testing on your own images. |
| ✅ **DONE** | **Reproduction scripts** | Training commands and inference examples are fully documented and tested. |
| 🔄 **IN PROGRESS** | **Pre‑trained weights (all variants)** | Official checkpoints for `adaptive`, `game2real`, `deformable`, `multiview`, and `panorama` are being open‑sourced. *ETA: within weeks.* |
| 🔄 **IN PROGRESS** | **Benchmark datasets** | Game character detection set, fisheye evaluation set, drone aerial set, and 360° panorama set are being prepared for public release under permissive licenses. *ETA: within weeks.* |
| ⏳ **TODO** | **ONNX / TensorRT export** | Production‑ready deployment scripts with INT8 calibration and end‑to‑end latency optimisation. |
| ⏳ **TODO** | **Colab tutorials** | Step‑by‑step notebooks for fine‑tuning on custom data and running inference on videos. |
| ⏳ **TODO** | **Hugging Face demo** | Online interactive demo integrated with 🤗 Spaces. |

> **Note:** Even without the official weights, you can train YOLO from scratch using the provided configs and your own dataset (e.g., COCO, VisDrone, or custom game screenshots). The codebase is fully functional and ready for research and development.

---

## 🏗️ Architecture Overview

```
Input → Scene Analysis → DomainAdaptiveLayer → ViewEmbedding →
DeformableA2C2f (×N) → DynamicScaleRouter → Detect(P3/P4/P5)
```

The pipeline consists of six stages:

1. **Scene Analysis** – lightweight heuristics classify the input scene type (game, fisheye, drone, panorama, standard).
2. **Adaptive Augmentation** (training only) – scene‑routed augmentation branches (game stylisation, fisheye distortion, perspective transform, domain mixup).
3. **Domain Adaptation** – `DomainAdaptiveLayer` with AdaIN aligns game→real feature statistics; `DomainAdversarialLoss` drives domain‑invariant learning via gradient reversal.
4. **Multi‑View Conditioning** – `ViewEmbedding` injects a learned 6‑class viewpoint embedding (pinhole, fisheye, panoramic, drone, BEV, ground).
5. **Deformable Feature Pyramid** – Deformable Area‑Attention + `DynamicScaleRouter` adapts sampling locations and scale weights per input.
6. **Detection Heads** – decoupled P3/P4/P5 heads with adaptive NMS.

---

## 🧩 Core Components

### 🔹 Deformable Area‑Attention (D‑AAttn)

Replaces standard area‑attention with a learnable 2D deformation field. The offset predictor warps the feature grid before computing attention, allowing the model to adapt to local geometric distortions.

| Module | Description |
|--------|-------------|
| `DeformableConv` | Dense warp‑then‑convolve; predicts per‑pixel offset field |
| `DeformableAAttn` | Area‑attention computed on a deformed grid |
| `DeformableA2C2f` | R‑ELAN block with deformable ABlocks |

**Complexity overhead:** only **+4.7%** parameters and **+4.1%** FLOPs per layer.

### 🔹 Game2Real Domain Adaptation

Three complementary mechanisms bridge the game‑rendering domain to the photographic domain:

- **Data‑level:** `GameCharacterStylization` applies posterisation (bit depth 3–6), unsharp masking, saturation boost (×1.5–1.8), and contrast adjustment.
- **Feature‑level:** `DomainAdaptiveLayer` uses Adaptive Instance Normalisation (AdaIN) to shift game‑domain feature statistics toward the real‑domain distribution.
- **Objective‑level:** `DomainAdversarialLoss` pits a domain classifier against the feature extractor in a minimax game.

**Ablation breakdown** (YOLOv12s baseline: 24.1 mAP on Game):
- +GameCharStylization: +11.7 mAP
- +DomainAdaptiveLayer: +6.5 mAP
- +DomainAdversarialLoss: +7.3 mAP

### 🔹 Multi‑View Conditioning

`ViewEmbedding` injects a learned 6‑class embedding (pinhole=0, fisheye=1, panoramic=2, drone=3, bev=4, ground=5) into backbone features via concatenation and 1×1 projection. `CrossViewConsistencyLoss` (NT‑Xent contrastive) pulls same‑class features from different views closer in embedding space.

**Theoretical guarantee:** Minimising $\mathcal{L}_{\text{cross}}$ bounds the $\mathcal{H}\Delta\mathcal{H}$‑distance between view‑specific distributions.

### 🔹 Adaptive Augmentation & Dynamic Routing

- **AdaptiveAugmentPolicy** – analyses each input via edge density, saturation mean, and contrast variance heuristics, then selects the optimal augmentation branch.
- **DynamicScaleRouter** – a lightweight gating network (1.8K params, 0.06 ms) that learns per‑input scale importance weights for P3/P4/P5.

### 🔹 Panoramic‑Specific Modules

- **CircularConv** – circular padding replaces zero‑padding in the horizontal dimension, connecting $x=W-1$ to $x=0$.
- **SphereAAttn** – partitions the feature map into latitude bands; equatorial bands receive proportionally more capacity than polar bands.

---

## 📦 Model Variants

| Variant | Key Modules | Target Scenario |
|---------|-------------|-----------------|
| `yolov14-deformable.yaml` | DeformableA2C2f | Fisheye / wide‑angle |
| `yolov14-multiview.yaml` | ViewEmbedding + CrossViewLoss | Drone / BEV / mixed perspectives |
| `yolov14-panorama.yaml` | SphereAAttn + CircularConv | 360° equirectangular |
| `yolov14-game2real.yaml` | DomainAdaptiveLayer + DomainAdvLoss | Game character detection |
| `yolov14-adaptive.yaml` | All components combined | Universal – auto scene detection |

---

## 🚀 Quick Start

```bash
conda create -n yolov14 python=3.11
conda activate yolov14
pip install -r requirements.txt
pip install -e .
```

**Train Game2Real model:**
```python
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v14/yolov14-game2real.yaml")
model.train(data="coco.yaml", epochs=300, imgsz=640)
```

**Train Adaptive model (all innovations):**
```python
model = YOLO("ultralytics/cfg/models/v14/yolov14-adaptive.yaml")
model.train(data="coco.yaml", epochs=300, imgsz=640)
```

**Inference – game characters detected as "person":**
```python
results = model.predict("delta_force_screenshot.jpg")
results[0].show()
```

**Web demo:**
```bash
python app.py
# Visit http://127.0.0.1:7860
```

---

## 📝 Citation

```bibtex
@article{jia2026yolov14,
  title={YOLOv14: Unified Cross-Domain Real-Time Object Detection with Adaptive Multi-View Representation},
  author={Jia, Jinling and Lu, Jian and Yawl, Jone and Zhang, Chenbin},
  journal={arXiv preprint arXiv:2608.04720},
  year={2026}
}
```

---

## 📄 License

[AGPL-3.0](LICENSE)

---

<p align="center">
  <strong>Built for researchers & developers who push object detection beyond ideal conditions.</strong>
  <br>
  <sub>⭐ If this project helps you, please give us a star!</sub>
</p>
