<h1 align="center" style="color:#1F4E79;">YOLO‑Omni: Omni‑Distortion Adaptive Routing with Sparse Target‑Prior for Real‑Time Object Detection</h1>
<p align="center">
  <a href="https://baike.baidu.com"><img src="https://img.shields.io/badge/Corresponding_Author-Z637826-green?style=flat-square" alt="Corresponding Author"></a>
</p>
<p align="center">
  <strong>The only real‑time detector that exceeds 43 mAP on all four challenging benchmarks – game, fisheye, drone, and panorama – simultaneously.</strong>
</p>

<div align="center" style="background:#EAF2FD; border:1px solid #4A90D9; border-radius:8px; padding:10px 14px;">
<strong style="color:#1F4E79;">Note:</strong> We name our detector YOLO‑Omni to highlight its capability for omni‑distortion robust detection across fisheye, panorama, aerial and game‑rendered domains.
</div>

<div align="center" style="background:#FDF3E7; border:1px solid #E5A940; border-radius:8px; padding:10px 14px;">
<strong style="color:#1F4E79;">Congratulations to the teachers and students of Harbin Media Vocational College</strong> on winning Third Prize in the New-Generation Information Technology track of the Provincial Vocational College Skills Competition! Their project tackled real corn-planting challenges, using <strong>YOLO-Omni</strong> (this project) as the core deep-learning algorithm for disease recognition, precise localization, and multi-terminal deployment, and was completed steadily through the 60-minute timed contest and technical defense. Read the full report <a href="https://mp.weixin.qq.com/s?__biz=Mzg2MTE4NDc0NA==&mid=2247519938&idx=1&sn=45d576efa2209ec44234cc2355b5cbef">here</a>.
</div>

---

<h2 id="table-of-contents" style="background:#1F4E79; color:#ffffff; padding:8px 14px; border-radius:6px;">Table of Contents</h2>

- [Why YOLO-Omni?](#why-yolo-omni)
- [Performance Highlights](#performance-highlights)
- [Project Roadmap & Status](#project-roadmap--status)
- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Model Variants](#model-variants)
- [Quick Start](#quick-start)
- [License](#license)

---

<h2 id="why-yolo-omni" style="background:#1F4E79; color:#ffffff; padding:8px 14px; border-radius:6px;">Why YOLO-Omni?</h2>

Conventional detectors excel under ideal pinhole‑camera conditions, but degrade sharply in **real‑world non‑ideal imaging** scenarios. YOLO-Omni learns **domain‑invariant, viewpoint‑robust** features via a combination of deformable attention, adaptive instance normalisation, and adversarial domain alignment:

| Scenario | Problem | YOLO-Omni Solution |
|----------|---------|------------------|
| **Fisheye / wide‑angle** | Barrel distortion shifts and compresses objects near edges | Deformable Area‑Attention (D‑AAttn) warps the feature grid to compensate for distortion |
| **Game footage** (Delta Force, COD, PUBG) | Rendering style (posterisation, edge sharpening, high saturation) causes missed detections | Game2Real domain adaptation with AdaIN + adversarial domain classifier aligns feature distributions |
| **Drone / top‑down view** | Unfamiliar scales and viewpoints, dense small objects | Multi‑view conditioning (ViewEmbedding) adapts to aerial perspectives |
| **360° panoramas** | Latitude stretching and 0°/360° boundary discontinuity | Spherical Attention (SphereAAttn) + CircularConv handle equirectangular projection |

---

<h2 id="performance-highlights" style="background:#1F4E79; color:#ffffff; padding:8px 14px; border-radius:6px;">Performance Highlights</h2>

| Metric | Value |
|--------|-------|
| **COCO mAP** (val2017, s‑scale) | **49.1** |
| **Latency** (T4 TensorRT FP16) | **2.91 ms** |
| **Throughput** | **344 FPS** |
| **Game benchmark** | **50.2 mAP** (+26.1 ↑ over YOLOv12s) |
| **Panorama benchmark** | **45.1 mAP** (+6.6 ↑ over best baseline) |
| **Drone benchmark** | **43.2 mAP** (+6.4 ↑ over best baseline) |
| **Fisheye benchmark** | **45.3 mAP** (+4.1 ↑ over best baseline) |

> **YOLO-Omni is the only real‑time detector that exceeds 43 mAP on all four challenging benchmarks at the same time.**

---

<h2 id="project-roadmap--status" style="background:#1F4E79; color:#ffffff; padding:8px 14px; border-radius:6px;">Project Roadmap & Status</h2>

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
| ⏳ **TODO** | **Hugging Face demo** | Online interactive demo integrated with Spaces. |

> **Note:** Even without the official weights, you can train YOLO-Omni from scratch using the provided configs and your own dataset (e.g., COCO, VisDrone, or custom game screenshots). The codebase is fully functional and ready for research and development.

---

<h2 id="architecture-overview" style="background:#1F4E79; color:#ffffff; padding:8px 14px; border-radius:6px;">Architecture Overview</h2>

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

<h2 id="core-components" style="background:#1F4E79; color:#ffffff; padding:8px 14px; border-radius:6px;">Core Components</h2>

<h3 style="color:#1F4E79; border-left:4px solid #1F4E79; padding-left:10px;">Deformable Area‑Attention (D‑AAttn)</h3>

Replaces standard area‑attention with a learnable 2D deformation field. The offset predictor warps the feature grid before computing attention, allowing the model to adapt to local geometric distortions.

| Module | Description |
|--------|-------------|
| `DeformableConv` | Dense warp‑then‑convolve; predicts per‑pixel offset field |
| `DeformableAAttn` | Area‑attention computed on a deformed grid |
| `DeformableA2C2f` | R‑ELAN block with deformable ABlocks |

**Complexity overhead:** only **+4.7%** parameters and **+4.1%** FLOPs per layer.

<h3 style="color:#1F4E79; border-left:4px solid #1F4E79; padding-left:10px;">Game2Real Domain Adaptation</h3>

Three complementary mechanisms bridge the game‑rendering domain to the photographic domain:

- **Data‑level:** `GameCharacterStylization` applies posterisation (bit depth 3–6), unsharp masking, saturation boost (×1.5–1.8), and contrast adjustment.
- **Feature‑level:** `DomainAdaptiveLayer` uses Adaptive Instance Normalisation (AdaIN) to shift game‑domain feature statistics toward the real‑domain distribution.
- **Objective‑level:** `DomainAdversarialLoss` pits a domain classifier against the feature extractor in a minimax game.

**Ablation breakdown** (YOLOv12s baseline: 24.1 mAP on Game):
- +GameCharStylization: +11.7 mAP
- +DomainAdaptiveLayer: +6.5 mAP
- +DomainAdversarialLoss: +7.3 mAP

<h3 style="color:#1F4E79; border-left:4px solid #1F4E79; padding-left:10px;">Multi‑View Conditioning</h3>

`ViewEmbedding` injects a learned 6‑class embedding (pinhole=0, fisheye=1, panoramic=2, drone=3, bev=4, ground=5) into backbone features via concatenation and 1×1 projection. `CrossViewConsistencyLoss` (NT‑Xent contrastive) pulls same‑class features from different views closer in embedding space.

**Theoretical guarantee:** Minimising $\mathcal{L}_{\text{cross}}$ bounds the $\mathcal{H}\Delta\mathcal{H}$‑distance between view‑specific distributions.

<h3 style="color:#1F4E79; border-left:4px solid #1F4E79; padding-left:10px;">Adaptive Augmentation & Dynamic Routing</h3>

- **AdaptiveAugmentPolicy** – analyses each input via edge density, saturation mean, and contrast variance heuristics, then selects the optimal augmentation branch.
- **DynamicScaleRouter** – a lightweight gating network (1.8K params, 0.06 ms) that learns per‑input scale importance weights for P3/P4/P5.

<h3 style="color:#1F4E79; border-left:4px solid #1F4E79; padding-left:10px;">Panoramic‑Specific Modules</h3>

- **CircularConv** – circular padding replaces zero‑padding in the horizontal dimension, connecting $x=W-1$ to $x=0$.
- **SphereAAttn** – partitions the feature map into latitude bands; equatorial bands receive proportionally more capacity than polar bands.

---

<h2 id="model-variants" style="background:#1F4E79; color:#ffffff; padding:8px 14px; border-radius:6px;">Model Variants</h2>

| Variant | Key Modules | Target Scenario |
|---------|-------------|-----------------|
| `yolo-deformable.yaml` | DeformableA2C2f | Fisheye / wide‑angle |
| `yolo-multiview.yaml` | ViewEmbedding + CrossViewLoss | Drone / BEV / mixed perspectives |
| `yolo-panorama.yaml` | SphereAAttn + CircularConv | 360° equirectangular |
| `yolo-game2real.yaml` | DomainAdaptiveLayer + DomainAdvLoss | Game character detection |
| `yolo-adaptive.yaml` | All components combined | Universal – auto scene detection |

---

<h2 id="quick-start" style="background:#1F4E79; color:#ffffff; padding:8px 14px; border-radius:6px;">Quick Start</h2>

```bash
conda create -n yoloomni python=3.11
conda activate yoloomni
pip install -r requirements.txt
pip install -e .
```

**Train Game2Real model:**
```python
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/yolo-game2real.yaml")
model.train(data="coco.yaml", epochs=300, imgsz=640)
```

**Train YOLO-Omni model (all innovations):**
```python
model = YOLO("ultralytics/cfg/models/yolo-adaptive.yaml")
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

<h2 id="license" style="background:#1F4E79; color:#ffffff; padding:8px 14px; border-radius:6px;">License</h2>

[AGPL-3.0](LICENSE)

---

<p align="center">
  <strong>Built for researchers & developers who push object detection beyond ideal conditions.</strong>
  <br>
  <sub>⭐ If this project helps you, please give us a star!</sub>
</p>
