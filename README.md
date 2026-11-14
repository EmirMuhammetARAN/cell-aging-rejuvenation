# Bidirectional Cellular Aging & Rejuvenation Morphology Translation (Bright-Field MSCs)

Code and experimental suite accompanying:

> **Bidirectional Young-to-Senescent and Senescent-to-Young-Like Morphology Translation in Bright-Field MSC Images: A Comparison of Latent Diffusion, CycleGAN, and CUTGAN**  
> Emir Muhammet ARAN — Department of Computer Engineering, Gazi University

---

## Overview

Chemical senescence assays such as SA-β-gal staining typically require fixation, preventing longitudinal observation of the same viable cells. This repository implements and benchmarks three generative image-translation families for bidirectional morphological simulation (`Young ↔ Senescent`) on bright-field mesenchymal stem cell (MSC) crops (`512×512`):

- **Latent Diffusion Model (`ldm_v12_l1`) + SDEdit + Classifier-Free Guidance (CFG):** Two-stage U-Net training on top of a frozen Stable Diffusion VAE (250 epochs MSE baseline + 50 epochs latent-space L1 structural fine-tuning, 15% unconditional dropout).
- **CUT (`CUT` / Contrastive Unpaired Translation):** PatchNCE-based unpaired image translation baseline.
- **CycleGAN:** ResNet-generator cycle-consistent GAN baseline.
- **Downstream Morphological & XAI Evaluation:** Auxiliary ResNet-18 classifier + Grad-CAM/LayerCAM attribution (`src/evaluation/`, `src/inference/gradcam/`) and Mask R-CNN morphological measurements (`cell area`, `circularity`, `aspect ratio`, `eccentricity`).

> **Scope & Biological Validity Disclaimer:** All comparative metrics reflect image-level morphological consistency and feature-distribution alignment (ImageNet FID) on an exploratory evaluation cohort; they do **not** establish biological rejuvenation or single-cell longitudinal viability.

---

## Dataset & Exploratory Benchmark Summary

- **Training Set:** `8,312` single-cell bright-field crops (`5,181` young, `3,131` senescent), combining manual annotations and Mask R-CNN pseudo-labels.
- **Evaluation Cohort:** `645` images (`327` young, `318` senescent), excluded from training and pseudo-labeling, reused across checkpoint and inference-parameter sweeps.

Results on the `645`-image evaluation cohort (per `bioRxiv.md`):

| Model / Checkpoint | Aging FID ↓ | Rejuvenation FID ↓ | Aging Target Rate (ResNet-18) | Rejuvenation Target Rate (ResNet-18) |
|---|---:|---:|---:|---:|
| **CUT (Epoch 200)** | **43.13** | **48.58** | 72.48% | **94.65%** |
| **CycleGAN (Epoch 160)** | 53.52 | 63.77 | 86.24% | 71.70% |
| **LDM `ldm_v12_l1` (Epoch 300, 5 seeds)** | 48.81 ± 0.99 | 54.00 ± 0.54 | **89.36% ± 0.55%** | 87.92% ± 1.27% |

CycleGAN exhibited strong training instability across epochs (e.g., aging FID swinging from `53.70` at epoch 160 to `272.05` at epoch 180 despite `99.69%` classifier prediction rate), whereas the LDM SDEdit pipeline achieved balanced bidirectional morphology conversion across seeds.

---

## Repository Structure

```text
cell-aging-rejuvenation/
├── bioRxiv.md                      # Preprint manuscript text & full statistical analysis
├── models/
│   ├── classifier/                 # ResNet-18 classifier & passage clock
│   ├── cut/                        # CUT generator & PatchNCE loss
│   ├── cyclegan/                   # CycleGAN ResNet generator & PatchGAN discriminator
│   └── ldm/                        # CellLDM, LPIPS/L1 datasets & v16 passage modules
├── src/
│   ├── data/                       # Parsing, cropping, v5 dataset preparation
│   ├── training/                   # LDM, CUT, CycleGAN, and classifier training scripts
│   ├── inference/                  # FID/metrics, epoch sweeps, seed invariance, Grad-CAM
│   ├── evaluation/                 # Mask R-CNN morphometrics & mask-guided SDEdit shields
│   └── demo/
│       └── app.py                  # Interactive Gradio UI (Translation, Grad-CAM, Sampling)
├── requirements.txt
└── setup.py
```

---

## Installation & Interactive Demo

```bash
git clone https://github.com/EmirMuhammetARAN/cell-aging-rejuvenation.git
cd cell-aging-rejuvenation
pip install -r requirements.txt
pip install -e .
```

Launch the Gradio demo (`src/demo/app.py`):

```bash
python src/demo/app.py
```

*Note:* Trained checkpoints (`checkpoints/ldm/checkpoint_v12_v5_data_lpips_last.pt`, `checkpoints/classifier/classifier_v2_5_resnet.pth`) and microscopy crops (`data/processed_v5/test/`) are excluded from Git tracking via `.gitignore`. Provide those files locally before launching the demo or evaluation scripts.

---

## License

See [LICENSE](LICENSE).
