# cell-aging-rejuvenation
Deep learning pipeline for simulating MSC cellular senescence and rejuvenation using CycleGAN and Latent Diffusion Models with LoRA fine-tuning.

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/EmirMuhammetARAN/cell-aging-rejuvenation.git
cd dataset_senescence
```

### 2. Install as a package (Recommended)
```bash
# Development mode (editable install)
pip install -e .
```

Or install dependencies manually:
```bash
pip install -r requirements.txt
```

### 3. Run scripts from root directory
```bash
# IMPORTANT: Always run from project root directory
cd /path/to/dataset_senescence

# Training
python src/training/cyclegan/train_cyclegan.py
python src/training/classifier/train_classifier.py
python src/training/ldm/train_ldm.py

# Inference
python src/inference/cyclegan/generate_all.py
python src/inference/classifier/evaluate_classifier.py
```

## Project Structure

```
models/              # Model architectures (reusable)
├── cyclegan/       # CycleGAN generator, discriminator
├── classifier/     # Image classifier
└── ldm/            # Latent Diffusion Model

src/
├── data/           # Data utilities & loaders
├── training/       # Training scripts
│   ├── cyclegan/
│   ├── classifier/
│   └── ldm/
├── inference/      # Inference & evaluation scripts
│   ├── cyclegan/
│   ├── classifier/
│   └── ldm/
└── evaluation/     # Evaluation metrics

data/               # Dataset
├── processed/      # Processed images
│   ├── train/
│   └── test/
└── raw/           # Raw images

checkpoints/        # Saved model weights
├── cyclegan/
├── classifier/
└── ldm/

results/            # Generated outputs
```

## Training

### CycleGAN
```bash
python src/training/cyclegan/train_cyclegan.py
```

### Image Classifier
```bash
python src/training/classifier/train_classifier.py
```

### Latent Diffusion Model
```bash
python src/training/ldm/train_ldm.py
```

## Inference

### Generate aged/rejuvenated images
```bash
python src/inference/cyclegan/generate_all.py
```

### Evaluate classifier
```bash
python src/inference/classifier/evaluate_classifier.py
```

## Troubleshooting

### Import errors
If you get `ModuleNotFoundError`, ensure:
1. You're running from the project root directory
2. You've installed the package: `pip install -e .`
3. Or set PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

### CUDA/GPU issues
Check if CUDA is available:
```python
import torch
print(torch.cuda.is_available())
```

### Missing data
Ensure data structure is set up correctly in `data/processed/train/young`, `data/processed/train/senescent`, etc.
