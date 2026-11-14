"""Unified multi-task PassageClock (Classifier v3.5) loader and inference utilities."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

DEFAULT_MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
DEFAULT_IMG_SIZE = 256
PASSAGE_MIN = 1.0
PASSAGE_SPAN = 39.0

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def get_default_v3_5_ckpt(root_dir: str) -> str:
    return os.path.join(root_dir, "checkpoints", "classifier", "classifier_v3_5_convnext.pth")


def uses_channels_last(model_name: str) -> bool:
    lower = model_name.lower()
    return not ("dinov2" in lower or "swin" in lower or "vit" in lower)


class PassageClock(nn.Module):
    """Multi-task continuous passage regression + 3-class phase classifier."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
        head_dropout: float = 0.3,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = int(img_size)

        kwargs = {"pretrained": pretrained, "num_classes": 0}
        if drop_path_rate > 0:
            kwargs["drop_path_rate"] = drop_path_rate
        lower = model_name.lower()
        if "dinov2" in lower or "swin" in lower or "vit" in lower:
            kwargs["img_size"] = self.img_size

        self.backbone = timm.create_model(model_name, **kwargs)
        feat_dim = self.backbone.num_features

        self.shared_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(head_dropout * 0.5),
        )
        self.age_head = nn.Linear(256, 1)
        self.phase_head = nn.Linear(256, 3)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        h = self.shared_head(feat)
        age = self.age_head(h).squeeze(-1)
        phase = self.phase_head(h)
        return age, phase


def load_passage_clock(ckpt_path: str, device: str = "cuda"):
    """
    Loads a PassageClock checkpoint and returns (model, metadata_dict).
    Metadata carries 'model_name', 'img_size', 'passage_min', 'passage_span',
    'use_channels_last', and calibration fields (e.g., 'val_optimal_passage_thresh').
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", ckpt.get("backbone", DEFAULT_MODEL_NAME))
    img_size = int(ckpt.get("img_size", DEFAULT_IMG_SIZE))
    passage_min = float(ckpt.get("passage_min", PASSAGE_MIN))
    passage_span = float(ckpt.get("passage_span", PASSAGE_SPAN))
    head_dropout = float(ckpt.get("head_dropout", 0.3))

    model = PassageClock(
        model_name=model_name,
        pretrained=False,
        drop_path_rate=0.0,
        head_dropout=head_dropout,
        img_size=img_size,
    )
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    use_cl = uses_channels_last(model_name) and str(device).startswith("cuda")
    if use_cl:
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    model.eval()

    meta = {
        "model_name": model_name,
        "img_size": img_size,
        "passage_min": passage_min,
        "passage_span": passage_span,
        "use_channels_last": use_cl,
        "val_optimal_passage_thresh": ckpt.get("val_optimal_passage_thresh"),
        "val_optimal_norm_thresh": ckpt.get("val_optimal_norm_thresh"),
        "epoch": ckpt.get("epoch"),
        "hfb14_rho": ckpt.get("hfb14_rho"),
        "msc_val_auc_age": ckpt.get("msc_val_auc_age"),
    }
    return model, meta


@torch.no_grad()
def predict_passages(
    model: PassageClock,
    imgs_01: torch.Tensor,
    device: str = "cuda",
    bs: int = 48,
    meta: dict = None,
    return_raw: bool = False,
):
    """
    Runs PassageClock inference on a CPU/GPU tensor of [0, 1] images (N, 3, H, W).
    Enforces luminance grayscale projection, native checkpoint resolution, and
    exact (passage = pred * 39.0 + 1.0) linear scale.
    """
    img_size = int(meta["img_size"]) if meta and "img_size" in meta else int(getattr(model, "img_size", DEFAULT_IMG_SIZE))
    p_min = float(meta["passage_min"]) if meta and "passage_min" in meta else PASSAGE_MIN
    p_span = float(meta["passage_span"]) if meta and "passage_span" in meta else PASSAGE_SPAN
    use_cl = (
        bool(meta["use_channels_last"])
        if meta and "use_channels_last" in meta
        else (uses_channels_last(getattr(model, "model_name", DEFAULT_MODEL_NAME)) and str(device).startswith("cuda"))
    )

    mean_t = torch.tensor(IMNET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(IMNET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    model.eval()
    raw_ages = []
    phase_probs = []
    for i in range(0, len(imgs_01), bs):
        x = imgs_01[i:i + bs].to(device).float()
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            g = gray.expand(-1, 3, -1, -1)
        elif x.shape[1] == 1:
            g = x.expand(-1, 3, -1, -1)
        else:
            raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(x.shape)}")

        if g.shape[2] != img_size or g.shape[3] != img_size:
            g = F.interpolate(g, size=(img_size, img_size), mode="bilinear", antialias=True, align_corners=False)

        g = (g - mean_t) / std_t
        if use_cl:
            g = g.contiguous(memory_format=torch.channels_last)

        age_pred, phase_logits = model(g)
        raw_ages.append(age_pred.float().cpu().numpy())
        if return_raw:
            phase_probs.append(torch.softmax(phase_logits.float(), dim=1).cpu().numpy())

    raw_arr = np.concatenate(raw_ages) if raw_ages else np.empty((0,), dtype=np.float32)
    passages = raw_arr * p_span + p_min
    if return_raw:
        prob_arr = np.concatenate(phase_probs, axis=0) if phase_probs else np.empty((0, 3), dtype=np.float32)
        return passages, raw_arr, prob_arr
    return passages
