"""Post-training evaluation of Classifier v2.5 and Classifier v3.5 on validation and test splits."""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader

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
from models.classifier.passage_clock import get_default_v3_5_ckpt, load_passage_clock, predict_passages


@torch.no_grad()
def eval_resnet_split(model, split_dir: str, device: str, force_grayscale: bool = False, batch_size: int = 64):
    if force_grayscale:
        tfm = transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    ds = ImageFolder(root=split_dir, transform=tfm)
    assert ds.class_to_idx == {"senescent": 0, "young": 1}, f"Unexpected classes: {ds.class_to_idx}"
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs_young, all_preds, all_labels = [], [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, memory_format=(torch.channels_last if device == "cuda" else torch.contiguous_format))
        logits = model(imgs)
        probs = torch.softmax(logits.float(), dim=1)
        all_probs_young.extend(probs[:, 1].cpu().numpy().tolist())
        all_preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(lbls.numpy().tolist())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    y_prob_young = np.asarray(all_probs_young, dtype=np.float64)
    # AUC for detecting Senescent (label 0) via 1 - P(Young) == AUC for Young (label 1) via P(Young)
    auc = float(roc_auc_score(y_true, y_prob_young))
    pr_auc_sen = float(average_precision_score((y_true == 0).astype(int), 1.0 - y_prob_young))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    filenames = [os.path.basename(p) for p, _ in ds.samples]
    df_preds = pd.DataFrame({
        "filename": filenames,
        "true_label_idx": y_true,
        "true_class": ["senescent" if y == 0 else "young" for y in y_true],
        "pred_label_idx": y_pred,
        "pred_class": ["senescent" if y == 0 else "young" for y in y_pred],
        "prob_young": y_prob_young,
        "prob_senescent": 1.0 - y_prob_young,
    })

    metrics = {
        "n_total": int(len(y_true)),
        "n_senescent": int((y_true == 0).sum()),
        "n_young": int((y_true == 1).sum()),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": auc,
        "pr_auc_senescent": pr_auc_sen,
        "senescent_recall": float(cm[0, 0] / max(1, cm[0].sum())),
        "young_recall": float(cm[1, 1] / max(1, cm[1].sum())),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, df_preds


def load_msc_split_tensors(split_dir: str):
    paths, labels, filenames, class_names = [], [], [], []
    for cls_name, lbl in [("young", 0), ("senescent", 1)]:
        cdir = os.path.join(split_dir, cls_name)
        if not os.path.exists(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                paths.append(os.path.join(cdir, fn))
                labels.append(lbl)
                filenames.append(fn)
                class_names.append(cls_name)
    imgs = torch.stack([TF.to_tensor(Image.open(p).convert("RGB")) for p in paths])
    return imgs, np.asarray(labels, dtype=np.int64), filenames, class_names


def eval_clock_msc_split(clock_model, clock_meta: dict, split_dir: str, device: str, frozen_thresh: float):
    imgs, y_true, filenames, class_names = load_msc_split_tensors(split_dir)
    passages, raw_scores, phase_probs = predict_passages(
        clock_model, imgs, device=device, bs=48, meta=clock_meta, return_raw=True
    )

    auc_age = float(roc_auc_score(y_true, passages))
    auc_phase = float(roc_auc_score(y_true, phase_probs[:, 2] - phase_probs[:, 0]))
    preds_frozen = (passages >= frozen_thresh).astype(np.int64)
    acc_frozen = float(accuracy_score(y_true, preds_frozen))
    bal_frozen = float(balanced_accuracy_score(y_true, preds_frozen))

    y_p = passages[y_true == 0]
    s_p = passages[y_true == 1]
    u_stat, p_val = mannwhitneyu(s_p, y_p, alternative="greater")

    df_preds = pd.DataFrame({
        "filename": filenames,
        "true_class": class_names,
        "true_label": y_true,
        "norm_score": raw_scores,
        "predicted_eq_passage": passages,
        "phase_young_prob": phase_probs[:, 0],
        "phase_inter_prob": phase_probs[:, 1],
        "phase_senes_prob": phase_probs[:, 2],
        "pred_senescent_at_val_thresh": preds_frozen,
    })

    metrics = {
        "n_total": int(len(y_true)),
        "n_young": int((y_true == 0).sum()),
        "n_senescent": int((y_true == 1).sum()),
        "roc_auc_passage": auc_age,
        "roc_auc_phase": auc_phase,
        "frozen_val_passage_thresh": float(frozen_thresh),
        "acc_at_val_thresh": acc_frozen,
        "bal_acc_at_val_thresh": bal_frozen,
        "young_mean_passage": float(y_p.mean()),
        "young_std_passage": float(y_p.std()),
        "young_median_passage": float(np.median(y_p)),
        "senes_mean_passage": float(s_p.mean()),
        "senes_std_passage": float(s_p.std()),
        "senes_median_passage": float(np.median(s_p)),
        "delta_passage_mean": float(s_p.mean() - y_p.mean()),
        "delta_passage_median": float(np.median(s_p) - np.median(y_p)),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(p_val),
    }
    return metrics, df_preds


def eval_clock_hfb14(clock_model, clock_meta: dict, device: str):
    meta_csv = os.path.join(root_dir, "deneme", "processed_tiles", "metadata.csv")
    img_dir = os.path.join(root_dir, "deneme", "processed_tiles", "images")
    df = pd.read_csv(meta_csv)
    hfb = df[df["donor"] == "hFB14"].reset_index(drop=True)
    imgs = torch.stack([TF.to_tensor(Image.open(os.path.join(img_dir, f)).convert("RGB")) for f in hfb["tile_filename"]])
    preds = predict_passages(clock_model, imgs, device=device, bs=48, meta=clock_meta)
    trues = hfb["passage"].values.astype(np.float32)

    rho = float(spearmanr(trues, preds).correlation)
    pear = float(pearsonr(trues, preds).statistic)
    mae = float(np.mean(np.abs(trues - preds)))
    rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
    return {
        "n_tiles": int(len(trues)),
        "spearman_rho": rho,
        "pearson_r": pear,
        "mae_passages": mae,
        "rmse_passages": rmse,
    }, trues, preds


def make_summary_figure(res_df_v25_test, res_df_v35_test, hfb_true, hfb_pred, m_v25, m_v35, m_hfb, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=300)

    # Panel 1: ROC Curves on blinded v5 test set
    ax = axes[0]
    fpr2, tpr2, _ = roc_curve(res_df_v25_test["true_label_idx"] == 0, res_df_v25_test["prob_senescent"])
    fpr3, tpr3, _ = roc_curve(res_df_v35_test["true_label"], res_df_v35_test["predicted_eq_passage"])
    ax.plot(fpr2, tpr2, lw=2.3, color="#2563eb", label=f"v2.5 ResNet-18 (AUC={m_v25['roc_auc']:.3f})")
    ax.plot(fpr3, tpr3, lw=2.3, color="#dc2626", label=f"v3.5 PassageClock (AUC={m_v35['roc_auc_passage']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.2)
    ax.set_title(f"Blinded MSC Test ROC (N={m_v25['n_total']})", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    # Panel 2: v3.5 Passage distribution on blinded v5 test MSCs
    ax = axes[1]
    y_vals = res_df_v35_test.loc[res_df_v35_test["true_label"] == 0, "predicted_eq_passage"].values
    s_vals = res_df_v35_test.loc[res_df_v35_test["true_label"] == 1, "predicted_eq_passage"].values
    bins = np.linspace(min(y_vals.min(), s_vals.min()) - 0.5, max(y_vals.max(), s_vals.max()) + 0.5, 28)
    ax.hist(y_vals, bins=bins, alpha=0.65, color="#10b981", density=True, label=f"Young (P{y_vals.mean():.1f}±{y_vals.std():.1f})")
    ax.hist(s_vals, bins=bins, alpha=0.65, color="#ef4444", density=True, label=f"Senescent (P{s_vals.mean():.1f}±{s_vals.std():.1f})")
    ax.axvline(m_v35["frozen_val_passage_thresh"], color="#1e293b", ls="--", lw=1.8,
               label=f"Frozen Val Thresh (P{m_v35['frozen_val_passage_thresh']:.2f})")
    ax.set_title(f"v3.5 MSC Test Separation (Delta={m_v35['delta_passage_mean']:+.2f}P)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Passage")
    ax.set_ylabel("Density")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=8.5)

    # Panel 3: v3.5 on held-out Picard fibroblast donor hFB14
    ax = axes[2]
    df_p = pd.DataFrame({"true": hfb_true, "pred": hfb_pred})
    grp = df_p.groupby("true")["pred"].agg(["mean", "std"]).reset_index()
    ax.scatter(hfb_true + np.random.default_rng(42).uniform(-0.25, 0.25, len(hfb_true)),
               hfb_pred, s=8, alpha=0.22, color="#64748b", label="Tiles (N=2100)")
    ax.errorbar(grp["true"], grp["mean"], yerr=grp["std"], fmt="o-", color="#7c3aed", lw=2.0, capsize=3, label="Mean ± SD")
    lim = [0, max(hfb_true.max(), hfb_pred.max()) + 2]
    ax.plot(lim, lim, "k--", alpha=0.5, lw=1.2, label="Identity (y=x)")
    ax.set_title(f"hFB14 Held-Out Clock (rho={m_hfb['spearman_rho']:.3f}, MAE={m_hfb['mae_passages']:.2f}P)", fontsize=12, fontweight="bold")
    ax.set_xlabel("True Passage")
    ax.set_ylabel("Predicted Passage")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Final Evaluation of v2.5 & v3.5 Classifiers on processed_v5")
    p.add_argument("--data_dir", type=str, default=os.path.join(root_dir, "data", "processed_v5"))
    p.add_argument("--v25_ckpt", type=str, default=os.path.join(root_dir, "checkpoints", "classifier", "classifier_v2_5_resnet.pth"))
    p.add_argument("--v35_ckpt", type=str, default=get_default_v3_5_ckpt(root_dir))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")
    results_cls_dir = os.path.join(root_dir, "results", "classifier")
    os.makedirs(results_cls_dir, exist_ok=True)

    # 1. Classifier v2.5 ResNet-18
    print("\n" + "=" * 78)
    print("EVALUATING CLASSIFIER V2.5 (RESNET-18)")
    print("=" * 78)
    resnet = Classifier(output_size=2)
    resnet.load_state_dict(torch.load(args.v25_ckpt, map_location="cpu", weights_only=True))
    resnet = resnet.to(device, memory_format=(torch.channels_last if device == "cuda" else torch.contiguous_format)).eval()

    m_v25_val, _ = eval_resnet_split(resnet, val_dir, device, force_grayscale=False)
    m_v25_test, df_v25_test = eval_resnet_split(resnet, test_dir, device, force_grayscale=False)
    m_v25_test_gray, _ = eval_resnet_split(resnet, test_dir, device, force_grayscale=True)

    print(f"[v2.5 Val]        N={m_v25_val['n_total']} | Acc={m_v25_val['acc']*100:.2f}% | BalAcc={m_v25_val['bal_acc']*100:.2f}% | AUC={m_v25_val['roc_auc']:.4f}")
    print(f"[v2.5 Test RGB]   N={m_v25_test['n_total']} | Acc={m_v25_test['acc']*100:.2f}% | BalAcc={m_v25_test['bal_acc']*100:.2f}% | AUC={m_v25_test['roc_auc']:.4f} | SenRec={m_v25_test['senescent_recall']*100:.1f}% | YngRec={m_v25_test['young_recall']*100:.1f}%")
    print(f"[v2.5 Test Gray]  N={m_v25_test_gray['n_total']} | Acc={m_v25_test_gray['acc']*100:.2f}% | BalAcc={m_v25_test_gray['bal_acc']*100:.2f}% | AUC={m_v25_test_gray['roc_auc']:.4f}")

    # 2. Classifier v3.5 PassageClock
    print("\n" + "=" * 78)
    print("EVALUATING CLASSIFIER V3.5 (PASSAGE CLOCK)")
    print("=" * 78)
    clock, clock_meta = load_passage_clock(args.v35_ckpt, device=device)
    frozen_th = clock_meta.get("val_optimal_passage_thresh")
    if frozen_th is None:
        raise ValueError("Missing val_optimal_passage_thresh in v3.5 checkpoint metadata")

    m_hfb14, hfb_true, hfb_pred = eval_clock_hfb14(clock, clock_meta, device)
    m_v35_val, _ = eval_clock_msc_split(clock, clock_meta, val_dir, device, frozen_th)
    m_v35_test, df_v35_test = eval_clock_msc_split(clock, clock_meta, test_dir, device, frozen_th)

    print(f"[v3.5 hFB14]      N={m_hfb14['n_tiles']} | Spearman={m_hfb14['spearman_rho']:.4f} | Pearson={m_hfb14['pearson_r']:.4f} | MAE={m_hfb14['mae_passages']:.2f}P | RMSE={m_hfb14['rmse_passages']:.2f}P")
    print(f"[v3.5 Val]        N={m_v35_val['n_total']} | AUC_P={m_v35_val['roc_auc_passage']:.4f} | AUC_Ph={m_v35_val['roc_auc_phase']:.4f} | BalAcc@ValTh={m_v35_val['bal_acc_at_val_thresh']*100:.2f}% (Th=P{frozen_th:.2f})")
    print(
        f"[v3.5 Test]       N={m_v35_test['n_total']} | AUC_P={m_v35_test['roc_auc_passage']:.4f} | AUC_Ph={m_v35_test['roc_auc_phase']:.4f} | "
        f"Acc@ValTh={m_v35_test['acc_at_val_thresh']*100:.2f}% | BalAcc@ValTh={m_v35_test['bal_acc_at_val_thresh']*100:.2f}%\n"
        f"                  Young=P{m_v35_test['young_mean_passage']:.2f}+/-{m_v35_test['young_std_passage']:.2f} | "
        f"Senes=P{m_v35_test['senes_mean_passage']:.2f}+/-{m_v35_test['senes_std_passage']:.2f} | "
        f"Delta={m_v35_test['delta_passage_mean']:+.2f}P | U-test p={m_v35_test['mannwhitney_p']:.2e}"
    )

    # Save outputs
    v35_csv_root = os.path.join(root_dir, "results", "classifier_v3_5_test_predictions.csv")
    v25_csv = os.path.join(results_cls_dir, "classifier_v2_5_test_predictions.csv")
    summary_json = os.path.join(results_cls_dir, "classifiers_v5_evaluation_summary.json")
    fig_png = os.path.join(results_cls_dir, "classifiers_v5_evaluation_figure.png")

    df_v35_test.to_csv(v35_csv_root, index=False)
    df_v25_test.to_csv(v25_csv, index=False)

    summary = {
        "v2_5_resnet": {"val": m_v25_val, "test_rgb": m_v25_test, "test_gray": m_v25_test_gray},
        "v3_5_passage_clock": {
            "model_name": clock_meta.get("model_name"),
            "img_size": clock_meta.get("img_size"),
            "checkpoint_epoch": clock_meta.get("epoch"),
            "hfb14": m_hfb14,
            "val": m_v35_val,
            "test": m_v35_test,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_summary_figure(df_v25_test, df_v35_test, hfb_true, hfb_pred, m_v25_test, m_v35_test, m_hfb14, fig_png)
    print(f"\n[SAVED] {v35_csv_root}")
    print(f"[SAVED] {v25_csv}")
    print(f"[SAVED] {summary_json}")
    print(f"[SAVED] {fig_png}")


if __name__ == "__main__":
    main()
