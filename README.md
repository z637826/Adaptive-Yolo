
# YOLOv14: Unified Cross-Domain Real-Time Object Detection with Adaptive Multi-View Representation

> **arXiv**: [2608.04720](https://arxiv.org/abs/2608.04720) · **Paper**: [PDF](https://arxiv.org/pdf/2608.04720) · **License**: AGPL-3.0

**YOLOv14** is a unified real-time object detection framework designed for **non-ideal imaging conditions** that standard detectors fail on. Unlike conventional YOLO variants that assume ideal pinhole-camera input, YOLOv14 learns **domain-invariant, viewpoint-robust features** through a combination of deformable attention, adaptive instance normalization, and adversarial domain alignment.

---

## 🔍 Why YOLOv14?

Standard object detectors achieve remarkable accuracy under controlled conditions, yet degrade sharply on four practical scenarios that arise routinely in real-world deployments:

| Scenario | Problem | YOLOv14 Solution |
|----------|---------|------------------|
| **Fisheye / wide-angle** | Radial barrel distortion shifts and compresses objects near image boundaries | Deformable Area-Attention (D-AAttn) warps the feature grid to compensate for distortion |
| **Game characters** (Delta Force, COD, PUBG) | Game-engine rendering produces distinct visual properties (posterization, edge sharpening, saturation boost) | Game2Real Domain Adaptation aligns game/real feature distributions via AdaIN + adversarial confusion |
| **Drone / top-down view** | Objects appear at unfamiliar angles and scales | Multi-View Conditioning + ViewEmbedding adapt to aerial perspectives |
| **360° panoramas** | Latitude-dependent stretching and boundary discontinuity at 0°/360° | Spherical Attention + CircularConv handle equirectangular projection |

---

## 📊 Performance Highlights

| Metric | Value |
|--------|-------|
| **COCO mAP** (val2017) | **49.1** (s-scale) |
| **Latency** (T4 TensorRT FP16) | **2.91 ms** (s-scale) |
| **Throughput** | **344 FPS** (s-scale) |
| **Game benchmark** | **50.2 mAP** (+26.1 over YOLOv12s) |
| **Panorama** | **45.1 mAP** (+6.6 over best baseline) |
| **Drone** | **43.2 mAP** (+6.4 over best baseline) |
| **Fisheye** | **45.3 mAP** (+4.1 over best baseline) |

**YOLOv14 is the only detector that exceeds 43 mAP on all four challenging benchmarks simultaneously.**

---

## 🗺️ Project Roadmap & Status

> **Last Updated:** August 2026

| Status | Task | Description |
|--------|------|-------------|
| ✅ **DONE** | **arXiv Technical Report** | Full paper (2608.04720) released with all mathematical formulations, ablation studies, and benchmark comparisons. |
| ✅ **DONE** | **Codebase (Architecture & Modules)** | Complete training/inference pipeline — including `DeformableAAttn`, `DomainAdaptiveLayer`, `SphereAAttn`, `ViewEmbedding`, `DynamicScaleRouter`, and all YAML configs — is open-sourced. |
| ✅ **DONE** | **Local Web Demo** | `app.py` (Gradio/Streamlit) is available for immediate testing on your own images. |
| ✅ **DONE** | **Reproduction Scripts** | Training commands and inference examples are fully documented and tested. |
| 🔄 **IN PROGRESS** | **Pre-trained Weights (All Variants)** | We are currently open-sourcing the official checkpoints for `yolov14-adaptive`, `yolov14-game2real`, `yolov14-deformable`, `yolov14-multiview`, and `yolov14-panorama`. *ETA: within weeks.* |
| 🔄 **IN PROGRESS** | **Benchmark Datasets** | The game character detection set, fisheye evaluation set, drone aerial set, and 360° panorama set are being prepared for public release under permissive licenses. *ETA: within weeks.* |
| ⏳ **TODO** | **ONNX / TensorRT Export** | Production-ready deployment scripts with INT8 calibration and end-to-end latency optimization. |
| ⏳ **TODO** | **Colab Tutorials** | Step-by-step notebooks for fine-tuning on custom data and running inference on videos. |
| ⏳ **TODO** | **Hugging Face Demo** | Online interactive demo integrated with 🤗 Spaces. |

**Note:** Even without the official weights, you can train YOLOv14 from scratch using the provided configs and your own dataset (e.g., COCO, VisDrone, or custom game screenshots). The codebase is fully functional and ready for research and development.

---

## 🏗️ Architecture Overview

```
Input → Scene Analysis → DomainAdaptiveLayer → ViewEmbedding →
DeformableA2C2f (×N) → DynamicScaleRouter → Detect(P3/P4/P5)
```

The pipeline consists of six stages:

1. **Scene Analysis** — lightweight heuristics classify the input scene type (game, fisheye, drone, panorama, standard)
2. **Adaptive Augmentation** (training only) — scene-routed augmentation branches (game stylization, fisheye distortion, perspective transform, domain mixup)
3. **Domain Adaptation** — DomainAdaptiveLayer with AdaIN aligns game→real feature statistics; DomainAdversarialLoss drives domain-invariant learning via gradient reversal
4. **Multi-View Conditioning** — ViewEmbedding injects a learned 6-class viewpoint embedding (pinhole, fisheye, panoramic, drone, BEV, ground)
5. **Deformable Feature Pyramid** — Deformable Area-Attention + DynamicScaleRouter adapts sampling locations and scale weights per input
6. **Detection Heads** — decoupled P3/P4/P5 heads with adaptive NMS

---

## 🧩 Core Components

### Deformable Area-Attention (D-AAttn)

Replaces standard area-attention with a learnable 2D deformation field. The offset predictor warps the feature grid before computing attention, allowing the model to adapt to local geometric distortions.

| Module | Description |
|--------|-------------|
| `DeformableConv` | Dense warp-then-convolve; predicts per-pixel offset field |
| `DeformableAAttn` | Area-attention computed on a deformed grid |
| `DeformableA2C2f` | R-ELAN block with deformable ABlocks |

**Complexity**: Only **+4.7%** parameters and **+4.1%** FLOPs overhead per layer.

### Game2Real Domain Adaptation

Three complementary mechanisms bridging the game-rendering domain to the photographic domain:

- **Data-level:** `GameCharacterStylization` applies posterization (bit depth 3–6), unsharp masking, saturation boost (×1.5–1.8), and contrast adjustment
- **Feature-level:** `DomainAdaptiveLayer` uses Adaptive Instance Normalization (AdaIN) to shift game-domain feature statistics toward the real-domain distribution
- **Objective-level:** `DomainAdversarialLoss` pits a domain classifier against the feature extractor in a minimax game

**Ablation breakdown** (YOLOv12s baseline: 24.1 mAP on Game):
- +GameCharStylization: +11.7 mAP
- +DomainAdaptiveLayer: +6.5 mAP
- +DomainAdversarialLoss: +7.3 mAP

### Multi-View Conditioning

`ViewEmbedding` injects a learned 6-class embedding (pinhole=0, fisheye=1, panoramic=2, drone=3, bev=4, ground=5) into backbone features via concatenation and 1×1 projection. `CrossViewConsistencyLoss` (NT-Xent contrastive) pulls same-class features from different views closer in embedding space.

**Theoretical guarantee**: Minimizing $\mathcal{L}_{\text{cross}}$ bounds the $\mathcal{H}\Delta\mathcal{H}$-distance between view-specific distributions.

### Adaptive Augmentation & Dynamic Routing

- **AdaptiveAugmentPolicy:** Analyzes each input via edge density, saturation mean, and contrast variance heuristics, then selects the optimal augmentation branch
- **DynamicScaleRouter:** Lightweight gating network (1.8K params, 0.06 ms) learns per-input scale importance weights for P3/P4/P5

### Panoramic-Specific Modules

- **CircularConv:** Circular padding replaces zero-padding in the horizontal dimension, connecting $x=W-1$ to $x=0$
- **SphereAAttn:** Feature map partitioned into latitude bands; equatorial bands receive proportionally more capacity than polar bands

---

## 📦 Model Variants

| Variant | Key Modules | Target Scenario |
|---------|-------------|-----------------|
| `yolov14-deformable.yaml` | DeformableA2C2f | Fisheye / wide-angle |
| `yolov14-multiview.yaml` | ViewEmbedding + CrossViewLoss | Drone / BEV / mixed perspectives |
| `yolov14-panorama.yaml` | SphereAAttn + CircularConv | 360° equirectangular |
| `yolov14-game2real.yaml` | DomainAdaptiveLayer + DomainAdvLoss | Game character detection |
| `yolov14-adaptive.yaml` | All components combined | Universal — auto scene detection |

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

**Inference — game characters detected as person:**
```python
results = model.predict("delta_force_screenshot.jpg")
results[0].show()
```

**Web demo:**
```bash
python app.py
# http://127.0.0.1:7860
```

---

## 📁 Project Structure

```
yolo/
├── app.py                              # Web demo
├── pipeline.png                        # System pipeline figure
├── pipeline_prompt.txt                 # Figure generation prompt
├── pipeline_tikz.tex                   # Pipeline TikZ source
├── fig_domain_adapt.tex                # Domain adaptation TikZ figure
├── table_ablation.tex                  # LaTeX tables for paper
├── latex_guide.tex                     # Compilation guide
├── ultralytics/
│   ├── nn/modules/
│   │   ├── block.py                    # A2C2f, DeformableAAttn, DeformableA2C2f,
│   │   │                              # ViewEmbedding, DynamicScaleRouter,
│   │   │                              # SphereAAttn, DomainAdaptiveLayer
│   │   ├── conv.py                    # Conv, DeformableConv, CircularConv
│   │   └── __init__.py
│   ├── nn/tasks.py                    # Model registry
│   ├── data/augment.py                # GameCharacterStylization,
│   │                                  # AdaptiveAugmentPolicy, DomainMixup
│   ├── utils/loss.py                  # CrossViewConsistencyLoss,
│   │                                  # DomainAdversarialLoss
│   └── cfg/models/v14/               # YOLOv14 model configs
│       ├── yolov14-deformable.yaml
│       ├── yolov14-multiview.yaml
│       ├── yolov14-panorama.yaml
│       ├── yolov14-game2real.yaml
│       └── yolov14-adaptive.yaml
└── README.md
```

---

## 📝 Citation

```bibtex
@article{lu2026yolov14,
  title={YOLOv14: Unified Cross-Domain Real-Time Object Detection with Adaptive Multi-View Representation},
  author={Lu, Jian and Jia, Jinling and Yawl, Jone and Zhang, Chenbin},
  journal={arXiv preprint arXiv:2608.04720},
  year={2026}
}
```

---

## 📄 License

[AGPL-3.0](LICENSE)
```
