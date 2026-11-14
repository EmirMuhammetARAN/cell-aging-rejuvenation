"""Classifier v2.5 (ResNet-18 binary senescence classifier trained and validated on processed_v5)."""

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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

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

from models.classifier.classifier import Classifier

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RandomRot90:
    """Applies exact 0/90/180/270 degree rotation without interpolation or corner artifacts."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        if k == 1:
            return img.transpose(Image.Transpose.ROTATE_90)
        if k == 2:
            return img.transpose(Image.Transpose.ROTATE_180)
        if k == 3:
            return img.transpose(Image.Transpose.ROTATE_270)
        return img


def build_resnet18_binary(pretrained: bool = True) -> Classifier:
    """Instantiates Classifier(output_size=2) with verified ImageNet ResNet-18 backbone."""
    clf = Classifier(output_size=2)
    if pretrained:
        ref = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        ref_state = {k: v for k, v in ref.state_dict().items() if not k.startswith("fc.")}
        clf.model.load_state_dict(ref_state, strict=False)
    return clf


@torch.no_grad()
def update_ema(ema_params, mod_params, ema_buffers, mod_buffers, decay: float):
    for ep, mp in zip(ema_params, mod_params):
        ep.mul_(decay).add_(mp, alpha=1.0 - decay)
    for eb, mb in zip(ema_buffers, mod_buffers):
        if eb.dtype.is_floating_point:
            eb.mul_(decay).add_(mb, alpha=1.0 - decay)
        else:
            eb.copy_(mb)


@torch.no_grad()
def evaluate(model, loader, criterion, device, young_idx: int, amp_dtype):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs = []
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n_samples += bs
        all_probs.extend(probs[:, young_idx].cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    y_prob = np.asarray(all_probs, dtype=np.float64)

    acc = float(np.mean(y_true == y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    y_binary = (y_true == young_idx).astype(np.int64)
    auc = float(roc_auc_score(y_binary, y_prob))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    senes_recall = float(cm[0, 0] / max(1, cm[0].sum()))
    young_recall = float(cm[1, 1] / max(1, cm[1].sum()))

    return {
        "loss": total_loss / max(1, n_samples),
        "acc": acc,
        "bal_acc": bal_acc,
        "auc": auc,
        "senes_recall": senes_recall,
        "young_recall": young_recall,
        "composite": 0.5 * auc + 0.5 * bal_acc,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train classifier_v2_5_resnet on data/processed_v5")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--head_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=2e-2)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default=os.path.join(root_dir, "data", "processed_v5"))
    p.add_argument("--save_path", type=str, default=os.path.join(root_dir, "checkpoints", "classifier", "classifier_v2_5_resnet.pth"))
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    results_dir = os.path.join(root_dir, "results", "classifier")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.82, 1.0), ratio=(0.95, 1.05), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRot90(),
        transforms.RandomGrayscale(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.35, hue=0.08),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(root=train_dir, transform=train_transform)
    val_ds = ImageFolder(root=val_dir, transform=val_transform)
    assert train_ds.class_to_idx == {"senescent": 0, "young": 1}, f"Unexpected mapping: {train_ds.class_to_idx}"
    assert val_ds.class_to_idx == train_ds.class_to_idx

    targets = np.asarray(train_ds.targets, dtype=np.int64)
    counts = np.bincount(targets, minlength=2)
    class_weights = len(targets) / (2.0 * np.maximum(counts, 1))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    print("=" * 78)
    print("CLASSIFIER V2.5 (RESNET-18) TRAINING - DATA PROCESSED_V5 (ZERO TEST PEEKING)")
    print("=" * 78)
    print(f"Device: {device} | AMP: {amp_dtype} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"Train samples: {len(train_ds)} (senescent={counts[0]}, young={counts[1]}) | Class weights: {class_weights.round(3).tolist()}")
    print(f"Val samples:   {len(val_ds)} (plate-disjoint val set)")

    model = build_resnet18_binary(pretrained=True).to(device, memory_format=torch.channels_last)
    ema_model = copy.deepcopy(model).eval()
    ema_params = [p.data for p in ema_model.parameters()]
    mod_params = [p.data for p in model.parameters()]
    ema_buffers = list(ema_model.buffers())
    mod_buffers = list(model.buffers())

    head_param_ids = {id(p) for p in model.model.fc.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    head_params = list(model.model.fc.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(2 * steps_per_epoch, max(1, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    best_score = -1.0
    best_epoch = 0
    best_metrics = None
    best_source = "ema"
    patience_counter = 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                logits = model(images)
                loss = train_criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema_params, mod_params, ema_buffers, mod_buffers, args.ema_decay)

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            train_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            train_total += bs

        train_loss = running_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        raw_m = evaluate(model, val_loader, val_criterion, device, young_idx=1, amp_dtype=amp_dtype)
        ema_m = evaluate(ema_model, val_loader, val_criterion, device, young_idx=1, amp_dtype=amp_dtype)

        if ema_m["composite"] >= raw_m["composite"]:
            eval_m, eval_net, eval_src = ema_m, ema_model, "ema"
        else:
            eval_m, eval_net, eval_src = raw_m, model, "raw"

        is_best = eval_m["composite"] > best_score
        if is_best:
            best_score = eval_m["composite"]
            best_epoch = epoch
            best_metrics = eval_m
            best_source = eval_src
            patience_counter = 0
            # Pure state_dict for drop-in compatibility with Classifier(2) callers
            cpu_state = {k: v.detach().cpu().contiguous() for k, v in eval_net.state_dict().items()}
            torch.save(cpu_state, args.save_path)
            meta_path = args.save_path.replace(".pth", "_meta.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "source": eval_src,
                    "class_to_idx": train_ds.class_to_idx,
                    "val_metrics": eval_m,
                    "args": vars(args),
                },
                meta_path,
            )
            tag = f" [BEST:{eval_src}]"
        else:
            patience_counter += 1
            tag = f" (patience {patience_counter}/{args.patience})"

        ep_s = time.time() - t_ep
        print(
            f"Ep {epoch:02d}/{args.epochs:02d} [{ep_s:.1f}s] | "
            f"TrLoss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
            f"Val[{eval_src}] Loss {eval_m['loss']:.4f} Acc {eval_m['acc']*100:.2f}% "
            f"BalAcc {eval_m['bal_acc']*100:.2f}% AUC {eval_m['auc']:.4f} "
            f"(Sen {eval_m['senes_recall']*100:.1f}% / Yng {eval_m['young_recall']*100:.1f}%){tag}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_source": eval_src,
                "val_loss": eval_m["loss"],
                "val_acc": eval_m["acc"],
                "val_bal_acc": eval_m["bal_acc"],
                "val_auc": eval_m["auc"],
                "val_senes_recall": eval_m["senes_recall"],
                "val_young_recall": eval_m["young_recall"],
                "val_composite": eval_m["composite"],
                "ema_auc": ema_m["auc"],
                "raw_auc": raw_m["auc"],
            }
        )

        if patience_counter >= args.patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch} (patience={args.patience}).")
            break

    hist_csv = os.path.join(results_dir, "classifier_v2_5_resnet_history.csv")
    pd.DataFrame(history).to_csv(hist_csv, index=False)

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"[DONE] Training finished in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(
        f"Best Epoch: {best_epoch} ({best_source}) | "
        f"Val AUC={best_metrics['auc']:.4f} | Val BalAcc={best_metrics['bal_acc']*100:.2f}% | "
        f"Val Acc={best_metrics['acc']*100:.2f}%"
    )
    print(f"Checkpoint: {args.save_path}")
    print(f"History:    {hist_csv}")
    print("=" * 78)


if __name__ == "__main__":
    main()
