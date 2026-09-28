# -*- coding: utf-8 -*-
"""
Biological Validation Suite for In-Silico Cellular Aging & Rejuvenation
======================================================================
Unified 1-File-Per-Shield Architecture:

1. shield1_morphometry.py  : Morphometrics, Difference Maps, CDI & Population Manifold Alignment
2. shield2_texture.py      : Cytoplasmic Texture, Shannon Entropy & Haralick GLCM Contrast
3. shield3_layercam.py     : Hierarchical LayerCAM & Spatial Attention Alignment
4. shield_panels.py        : 6-Panel 300 DPI Publication Matrix (Exemplars vs Failure Modes)
5. seed_invariance.py      : Seed Invariance, Consensus & Parametric Stability (5 Seeds)
6. sampling_steps.py       : Sampling Steps Ablation, Multi-Step ODE Proof & Pseudo-Timelapse
7. mrcnn_mask_extractor.py : Mask R-CNN Single-Cell Segmentation Mask Extractor
"""

from .shield1_morphometry import compute_morphometrics, compute_difference_map
from .shield2_texture import compute_shannon_entropy, extract_cytoplasmic_texture
from .shield3_layercam import HierarchicalLayerCAM, compute_alignment_metrics, create_composite_overlay

__all__ = [
    'compute_morphometrics',
    'compute_difference_map',
    'compute_shannon_entropy',
    'extract_cytoplasmic_texture',
    'HierarchicalLayerCAM',
    'compute_alignment_metrics',
    'create_composite_overlay'
]
