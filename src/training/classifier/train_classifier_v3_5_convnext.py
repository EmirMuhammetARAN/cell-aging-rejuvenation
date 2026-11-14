"""Classifier v3.5 (ConvNeXt-Small IN22k multi-task senescence and passage clock)."""

import os
import sys
import copy
import math
import time
import random
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.classifier.passage_clock import (
    PassageClock,
    DEFAULT_MODEL_NAME,
    DEFAULT_IMG_SIZE,
    PASSAGE_MIN,
    PASSAGE_SPAN,
    IMNET_MEAN,
    IMNET_STD,
    uses_channels_last,
)

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

CACHE_SIZE = 288


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preload_grayscale_cache(paths):
    n = len(paths)
    buf = torch.empty((n, 1, CACHE_SIZE, CACHE_SIZE), dtype=torch.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im_g = im.convert("L").resize((CACHE_SIZE, CACHE_SIZE), Image.Resampling.BILINEAR)
            arr = np.frombuffer(im_g.tobytes(), dtype=np.uint8).reshape(CACHE_SIZE, CACHE_SIZE)
            buf[i, 0] = torch.from_numpy(arr.copy())
    return buf


@torch.no_grad()
def gpu_augment_batch(x_uint8, img_size: int, mean_t: torch.Tensor, std_t: torch.Tensor):
    b = x_uint8.shape[0]
    x = x_uint8.float().mul_(1.0 / 255.0)

    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])
    k = random.randint(0, 3)
    if k > 0:
        x = torch.rot90(x, k=k, dims=[2, 3])

    angles = (torch.rand(b, device=x.device) * 2 - 1) * math.pi
    scales = 0.85 + 0.30 * torch.rand(b, device=x.device)
    tx = (torch.rand(b, device=x.device) * 2 - 1) * 0.06
    ty = (torch.rand(b, device=x.device) * 2 - 1) * 0.06

    cos_a = torch.cos(angles) / scales
    sin_a = torch.sin(angles) / scales
    theta = torch.zeros((b, 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (b, 1, img_size, img_size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    contrast = 0.75 + 0.50 * torch.rand((b, 1, 1, 1), device=x.device)
    brightness = (torch.rand((b, 1, 1, 1), device=x.device) * 2 - 1) * 0.15
    mean_per_img = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - mean_per_img) * contrast + mean_per_img + brightness).clamp_(1e-4, 1.0)
    gamma = 0.80 + 0.45 * torch.rand((b, 1, 1, 1), device=x.device)
    x = torch.pow(x, gamma).clamp_(0.0, 1.0)
    if random.random() < 0.25:
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    x = x.expand(-1, 3, -1, -1)
    return (x - mean_t) / std_t


@torch.no_grad()
def gpu_eval_prep(x_uint8, img_size: int, mean_t: torch.Tensor, std_t: torch.Tensor):
    x = x_uint8.float().mul_(1.0 / 255.0)
    if x.shape[2] != img_size or x.shape[3] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = x.expand(-1, 3, -1, -1)
    return (x - mean_t) / std_t


@torch.no_grad()
def update_ema(ema_params, mod_params, ema_bufs, mod_bufs, decay: float):
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, mod_params, alpha=1.0 - decay)
    for eb, mb in zip(ema_bufs, mod_bufs):
        eb.copy_(mb)


def find_optimal_val_threshold(scores: np.ndarray, labels: np.ndarray):
    best_bal = -1.0
    best_acc = 0.0
    best_th = float(np.median(scores))
    for th in np.quantile(scores, np.linspace(0.05, 0.95, 181)):
        preds = (scores >= th).astype(np.int64)
        bal = balanced_accuracy_score(labels, preds)
        acc = accuracy_score(labels, preds)
        if bal > best_bal or (abs(bal - best_bal) < 1e-6 and acc > best_acc):
            best_bal = float(bal)
            best_acc = float(acc)
            best_th = float(th)
    return best_th, best_bal, best_acc


@torch.no_grad()
def evaluate_validation(
    model,
    hfb_buf,
    hfb_passages,
    msc_buf,
    msc_labels,
    img_size: int,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    use_cl: bool,
    amp_dtype,
    batch_size: int = 96,
):
    model.eval()
    hfb_preds = []
    for i in range(0, len(hfb_buf), batch_size):
        xb = gpu_eval_prep(hfb_buf[i:i + batch_size], img_size, mean_t, std_t)
        if use_cl:
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            age, _ = model(xb)
        hfb_preds.append(age.float().cpu().numpy())
    hfb_preds = np.concatenate(hfb_preds)
    pred_p = hfb_preds * PASSAGE_SPAN + PASSAGE_MIN
    rho = float(spearmanr(hfb_passages, pred_p).correlation)
    pear = float(pearsonr(hfb_passages, pred_p).statistic)
    mae = float(np.mean(np.abs(hfb_passages - pred_p)))

    msc_ages, msc_phase_scores = [], []
    for i in range(0, len(msc_buf), batch_size):
        xb = gpu_eval_prep(msc_buf[i:i + batch_size], img_size, mean_t, std_t)
        if use_cl:
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            age, phase = model(xb)
            probs = torch.softmax(phase.float(), dim=1)
        msc_ages.append(age.float().cpu().numpy())
        msc_phase_scores.append((probs[:, 2] - probs[:, 0]).cpu().numpy())

    msc_ages = np.concatenate(msc_ages)
    msc_phase_scores = np.concatenate(msc_phase_scores)
    auc_age = float(roc_auc_score(msc_labels, msc_ages))
    auc_phase = float(roc_auc_score(msc_labels, msc_phase_scores))

    norm_th, val_bal_acc, val_acc = find_optimal_val_threshold(msc_ages, msc_labels)
    passage_th = norm_th * PASSAGE_SPAN + PASSAGE_MIN

    return {
        "hfb14_rho": rho,
        "hfb14_pearson": pear,
        "hfb14_mae": mae,
        "msc_auc_age": auc_age,
        "msc_auc_phase": auc_phase,
        "msc_val_bal_acc": val_bal_acc,
        "msc_val_acc": val_acc,
        "val_norm_thresh": norm_th,
        "val_passage_thresh": passage_th,
        "composite": rho + auc_age,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train classifier_v3_5_convnext (ConvNeXt-Small IN22k)")
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--val_donor", type=str, default="hFB14")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--bb_lr", type=float, default=5e-5)
    p.add_argument("--head_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--drop_path", type=float, default=0.30)
    p.add_argument("--head_drop", type=float, default=0.30)
    p.add_argument("--ema_decay", type=float, default=0.992)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", type=str, default=os.path.join(root_dir, "checkpoints", "classifier", "classifier_v3_5_convnext.pth"))
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_cl = uses_channels_last(args.model_name) and device == "cuda"

    picard_meta = os.path.join(root_dir, "deneme", "processed_tiles", "metadata.csv")
    picard_dir = os.path.join(root_dir, "deneme", "processed_tiles", "images")
    msc_val_dir = os.path.join(root_dir, "data", "processed_v5", "val")
    cache_pt = os.path.join(root_dir, "scratch", "picard_v5val_cache_288.pt")
    results_dir = os.path.join(root_dir, "results", "classifier")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(cache_pt), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(picard_meta)
    phase_map = {"young": 0, "intermediate": 1, "senescent": 2}
    df["phase_id"] = df["age_group"].map(phase_map)
    tr_df = df[df["donor"] != args.val_donor].reset_index(drop=True)
    hfb_df = df[df["donor"] == args.val_donor].reset_index(drop=True)

    if os.path.exists(cache_pt):
        print(f"[1/3] Loading cached tensor bundle from {cache_pt}...")
        bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
        tr_buf = bundle["tr_buf"]
        hfb_buf = bundle["hfb_buf"]
        msc_buf = bundle["msc_buf"]
        msc_labels = bundle["msc_labels"]
    else:
        print("[1/3] Caching grayscale 288x288 images into CPU RAM (zero test peeking)...")
        tr_paths = [os.path.join(picard_dir, f) for f in tr_df["tile_filename"]]
        hfb_paths = [os.path.join(picard_dir, f) for f in hfb_df["tile_filename"]]
        msc_paths, msc_labels = [], []
        for cls_name, lbl in [("young", 0), ("senescent", 1)]:
            d = os.path.join(msc_val_dir, cls_name)
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
                    msc_paths.append(os.path.join(d, f))
                    msc_labels.append(lbl)
        msc_labels = np.asarray(msc_labels, dtype=np.int64)
        tr_buf = preload_grayscale_cache(tr_paths)
        hfb_buf = preload_grayscale_cache(hfb_paths)
        msc_buf = preload_grayscale_cache(msc_paths)
        torch.save({"tr_buf": tr_buf, "hfb_buf": hfb_buf, "msc_buf": msc_buf, "msc_labels": msc_labels}, cache_pt)

    if device == "cuda":
        tr_buf = tr_buf.pin_memory()
    tr_age = torch.tensor(tr_df["normalized_age"].values, dtype=torch.float32, device=device)
    tr_phase = torch.tensor(tr_df["phase_id"].values, dtype=torch.long, device=device)
    hfb_buf = hfb_buf.to(device)
    msc_buf = msc_buf.to(device)
    hfb_passages = hfb_df["passage"].values.astype(np.float32)

    mean_t = torch.tensor(IMNET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(IMNET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    print("=" * 80)
    print(f"CLASSIFIER V3.5 TRAINING ({args.model_name}, img={args.img_size})")
    print("=" * 80)
    print(f"Picard Train: {len(tr_df)} tiles ({tr_df['donor'].nunique()} donors) | Held-out {args.val_donor}: {len(hfb_df)} tiles")
    print(f"MSC Val:      {len(msc_labels)} cells from data/processed_v5/val (Zero Test Peeking!)")
    print(f"Batch Size:   {args.batch_size} | Epochs: {args.epochs} (Patience={args.patience}) | channels_last={use_cl}")

    model = PassageClock(
        model_name=args.model_name,
        pretrained=True,
        drop_path_rate=args.drop_path,
        head_dropout=args.head_drop,
        img_size=args.img_size,
    ).to(device)
    if use_cl:
        model = model.to(memory_format=torch.channels_last)
    ema_model = copy.deepcopy(model).eval()

    ema_params = [p.data for p in ema_model.parameters()]
    mod_params = [p.data for p in model.parameters()]
    ema_buffers = list(ema_model.buffers())
    mod_buffers = list(model.buffers())

    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.shared_head.parameters())
        + list(model.age_head.parameters())
        + list(model.phase_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.bb_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    n_tr = len(tr_buf)
    steps_per_epoch = math.ceil(n_tr / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, 1 * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    reg_crit = nn.SmoothL1Loss(beta=0.03)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_score = -1.0
    best_epoch = 0
    best_metrics = None
    best_src = "raw"
    patience_counter = 0
    history = []
    t_start = time.time()

    print("\n[2/3] Training PassageClock v3.5...")
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        model.train()
        perm_cpu = torch.randperm(n_tr)
        running_loss = 0.0

        for i in range(0, n_tr, args.batch_size):
            idx_cpu = perm_cpu[i:i + args.batch_size]
            idx_gpu = idx_cpu.to(device, non_blocking=True)
            xb_u8 = tr_buf[idx_cpu].to(device, non_blocking=True)
            xb = gpu_augment_batch(xb_u8, args.img_size, mean_t, std_t)
            if use_cl:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb_age = tr_age[idx_gpu]
            yb_phase = tr_phase[idx_gpu]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                pred_age, pred_phase = model(xb)
                loss = reg_crit(pred_age, yb_age) + 0.30 * cls_crit(pred_phase, yb_phase)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema_params, mod_params, ema_buffers, mod_buffers, decay=args.ema_decay)
            running_loss += float(loss.item())

        raw_m = evaluate_validation(
            model, hfb_buf, hfb_passages, msc_buf, msc_labels, args.img_size, mean_t, std_t, use_cl, amp_dtype
        )
        ema_m = evaluate_validation(
            ema_model, hfb_buf, hfb_passages, msc_buf, msc_labels, args.img_size, mean_t, std_t, use_cl, amp_dtype
        )
        if ema_m["composite"] >= raw_m["composite"]:
            eval_m, eval_net, eval_src = ema_m, ema_model, "ema"
        else:
            eval_m, eval_net, eval_src = raw_m, model, "raw"

        is_best = eval_m["composite"] > best_score
        if is_best:
            best_score = eval_m["composite"]
            best_epoch = epoch
            best_metrics = eval_m
            best_src = eval_src
            patience_counter = 0
            cpu_state = {k: v.detach().cpu().contiguous() for k, v in eval_net.state_dict().items()}
            torch.save(
                {
                    "epoch": epoch,
                    "source": eval_src,
                    "model_name": args.model_name,
                    "img_size": args.img_size,
                    "head_dropout": args.head_drop,
                    "passage_min": PASSAGE_MIN,
                    "passage_span": PASSAGE_SPAN,
                    "val_optimal_norm_thresh": eval_m["val_norm_thresh"],
                    "val_optimal_passage_thresh": eval_m["val_passage_thresh"],
                    "hfb14_rho": eval_m["hfb14_rho"],
                    "hfb14_mae": eval_m["hfb14_mae"],
                    "msc_val_auc_age": eval_m["msc_auc_age"],
                    "msc_val_auc_phase": eval_m["msc_auc_phase"],
                    "msc_val_bal_acc": eval_m["msc_val_bal_acc"],
                    "msc_val_acc": eval_m["msc_val_acc"],
                    "composite_score": eval_m["composite"],
                    "model_state_dict": cpu_state,
                    "args": vars(args),
                },
                args.save_path,
            )
            tag = f" [BEST:{eval_src}]"
        else:
            patience_counter += 1
            tag = f" (patience {patience_counter}/{args.patience})"

        torch.save(
            {
                "epoch": epoch,
                "model_name": args.model_name,
                "img_size": args.img_size,
                "head_dropout": args.head_drop,
                "passage_min": PASSAGE_MIN,
                "passage_span": PASSAGE_SPAN,
                "model_state_dict": {k: v.detach().cpu().contiguous() for k, v in eval_net.state_dict().items()},
            },
            args.save_path.replace(".pth", "_last.pth"),
        )

        ep_s = time.time() - te
        print(
            f"Ep {epoch:02d}/{args.epochs:02d} [{ep_s:.1f}s] [{eval_src}] | loss={running_loss/steps_per_epoch:.4f} | "
            f"hFB14 rho={eval_m['hfb14_rho']:.4f} MAE={eval_m['hfb14_mae']:.2f}P | "
            f"MSC AUC={eval_m['msc_auc_age']:.4f} BalAcc={eval_m['msc_val_bal_acc']*100:.2f}% "
            f"(th=P{eval_m['val_passage_thresh']:.2f}) | comp={eval_m['composite']:.4f}{tag}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / steps_per_epoch,
                "eval_src": eval_src,
                **eval_m,
                "raw_composite": raw_m["composite"],
                "ema_composite": ema_m["composite"],
            }
        )

        if patience_counter >= args.patience:
            print(f"\n[!] Early stopping triggered at Epoch {epoch} (patience={args.patience}).")
            break

    hist_csv = os.path.join(results_dir, "classifier_v3_5_convnext_history.csv")
    pd.DataFrame(history).to_csv(hist_csv, index=False)

    print("\n" + "=" * 80)
    print(f"[DONE] PassageClock v3.5 finished in {(time.time()-t_start)/60.0:.2f} min")
    print(
        f"Best Epoch: {best_epoch} ({best_src}) | hFB14 rho={best_metrics['hfb14_rho']:.4f} "
        f"MAE={best_metrics['hfb14_mae']:.2f}P | MSC Val AUC={best_metrics['msc_auc_age']:.4f} "
        f"BalAcc={best_metrics['msc_val_bal_acc']*100:.2f}% (Val Thresh=P{best_metrics['val_passage_thresh']:.2f})"
    )
    print(f"Checkpoint: {args.save_path}")
    print(f"History:    {hist_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
