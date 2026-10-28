"""
LDM v16 Stage-1 Evaluation: Does the passage condition actually control morphology?
===================================================================================
Judge: ConvNeXt-Tiny classifier v3 (deneme/checkpoints/classifier_v3_convnext_best.pth), trained on all
donors EXCEPT hFB14. All real reference data below is the held-out donor hFB14 (never seen by the
classifier nor by the LDM), so the judge is independent of the generator.

Parts
  0. Calibration  : classifier on REAL hFB14 tiles (predicted vs true passage) = the judge's own ceiling.
  A. Noise -> image sweep: identical noise seeds across target passages (paired), CFG sampling.
       -> predicted passage vs target passage, Spearman, monotonicity, slope.
  B. SDEdit aging / rejuvenation of real hFB14 tiles (young -> old targets, old -> young targets).
  C. FID per age group (young P<=8, intermediate 9-19, senescent >=20) vs real hFB14 tiles of that group.
  D. Shortcut check: mean/std pixel intensity of generated images vs target passage (cf. phase/batch confound).

Usage:
    python src/inference/ldm/eval_v16_passage_sweep.py                    # best_model, n=32/passage
    python src/inference/ldm/eval_v16_passage_sweep.py --ckpt checkpoints/ldm/checkpoint_v16_fibroblast_passage_epoch_30.pt
    python src/inference/ldm/eval_v16_passage_sweep.py --n_gen 16 --skip_fid   # quick check
"""

import argparse
import gc
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from torchvision import models
from torchvision.transforms import functional as TF
from torchvision.utils import save_image, make_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_v16 import CellLDM_v16

VAL_DONOR = 'hFB14'
TILES_DIR = os.path.join(root_dir, 'deneme', 'processed_tiles')
CLS_CKPT = os.path.join(root_dir, 'deneme', 'checkpoints', 'classifier_v3_convnext_best.pth')
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ConvNeXtMultiTask(nn.Module):
    """Same architecture as deneme/evaluate_convnext_test.py (age head = sigmoid of (p-1)/39)."""
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[2].in_features
        self.norm = base.classifier[0]
        self.age_head = nn.Sequential(nn.Flatten(), nn.Linear(in_features, 128), nn.LayerNorm(128),
                                      nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())
        self.phase_head = nn.Sequential(nn.Flatten(), nn.Linear(in_features, 128), nn.LayerNorm(128),
                                        nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        x = self.norm(self.avgpool(self.features(x)))
        return self.age_head(x).squeeze(1), self.phase_head(x)


@torch.no_grad()
def predict_passage(cls, imgs01, device, bs=32):
    """imgs01: (N,3,H,W) in [0,1] on CPU. Returns predicted passage (N,) = score*39+1."""
    out = []
    for i in range(0, len(imgs01), bs):
        x = imgs01[i:i + bs].to(device).float()
        g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).repeat(1, 3, 1, 1)
        g = F.interpolate(g, size=(256, 256), mode='bilinear', antialias=True, align_corners=False)
        g = (g - IMNET_MEAN.to(device)) / IMNET_STD.to(device)
        age, _ = cls(g)
        out.append((age.float().cpu().numpy() * 39.0 + 1.0))
    return np.concatenate(out)


def load_tile(fname):
    return TF.to_tensor(Image.open(os.path.join(TILES_DIR, 'images', fname)).convert('RGB'))


def monotone_stats(targets, means):
    targets, means = np.asarray(targets, float), np.asarray(means, float)
    diffs = np.diff(means)
    slope = float(np.polyfit(targets, means, 1)[0])
    rho = float(spearmanr(targets, means).correlation)
    return {'spearman_means': rho, 'frac_adjacent_increasing': float((diffs > 0).mean()), 'slope': slope}


def md_table(df):
    """Minimal markdown table (avoids the optional 'tabulate' dependency)."""
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in r.tolist()) + ' |')
    return '\n'.join(lines)


def stage_fid(gen_dir, ref_dir, device):
    from pytorch_fid import fid_score
    return float(fid_score.calculate_fid_given_paths([gen_dir, ref_dir], batch_size=32, device=device, dims=2048))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model_v16_fibroblast_passage.pt'))
    ap.add_argument('--out', default=os.path.join(root_dir, 'results', 'v16_evaluation'))
    ap.add_argument('--n_gen', type=int, default=32, help='images per target passage (Part A)')
    ap.add_argument('--n_src', type=int, default=16, help='source tiles per SDEdit direction (Part B)')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--cfg', type=float, default=3.0)
    ap.add_argument('--strength', type=float, default=0.7)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--skip_fid', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    print(f"\n{'='*80}\n  LDM v16 EVALUATION | ckpt={os.path.basename(args.ckpt)} | n_gen={args.n_gen} | "
          f"steps={args.steps} | cfg={args.cfg} | seed={args.seed}\n{'='*80}")

    # ---- models
    cls = ConvNeXtMultiTask().to(device)
    cls.load_state_dict(torch.load(CLS_CKPT, map_location=device)['model_state_dict'])
    cls.eval()

    model = CellLDM_v16()
    model.to(device)
    model.vae.to(memory_format=torch.channels_last)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dicts(ckpt)
    model.unet.eval(); model.cond.eval()
    print(f"[*] Loaded LDM epoch {ckpt.get('epoch', '?')} (val {ckpt.get('best_val_loss', float('nan')):.4f}) -> using EMA weights")
    del ckpt; gc.collect()

    meta = pd.read_csv(os.path.join(TILES_DIR, 'metadata.csv'))
    meta = meta[meta['donor'] == VAL_DONOR].reset_index(drop=True)

    # ---------------------------------------------------------------- Part 0: real calibration
    print("\n[0] Calibration: classifier on REAL held-out donor tiles")
    cal = meta.iloc[::3].reset_index(drop=True)
    real_imgs = torch.stack([load_tile(f) for f in cal['tile_filename']])
    cal['pred'] = predict_passage(cls, real_imgs, device)
    real_rho = float(spearmanr(cal['passage'], cal['pred']).correlation)
    real_mae = float((cal['pred'] - cal['passage']).abs().mean())
    cal_by_p = cal.groupby('passage')['pred'].mean()
    print(f"    real hFB14: Spearman(true,pred)={real_rho:.3f} | MAE={real_mae:.2f} passages | N={len(cal)}")
    cal_means_by_p = cal_by_p.to_dict()
    real_intensity = float(real_imgs.mean())
    real_int_by_group = {}
    _pp = cal['passage'].to_numpy()
    for _g, _m in (('young(P<=8)', _pp <= 8), ('inter(P9-19)', (_pp > 8) & (_pp < 20)), ('senes(P>=20)', _pp >= 20)):
        real_int_by_group[_g] = float(real_imgs[torch.from_numpy(_m)].mean()) if _m.any() else float('nan')
    print(f"    real hFB14 mean intensity by group: " + ' | '.join(f'{k}={v:.3f}' for k, v in real_int_by_group.items()))

    # ---------------------------------------------------------------- Part A: noise -> image sweep
    print("\n[A] Paired noise->image passage sweep")
    targets = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    n = args.n_gen
    rows_a, gen_by_p, bright = [], {}, {}
    grid_imgs = {}
    for tp in targets:
        imgs = []
        for s in range(0, n, args.batch):
            idxs = list(range(s, min(n, s + args.batch)))
            noise = torch.cat([torch.randn(1, 4, 64, 64, device=device,
                                           generator=torch.Generator(device=device).manual_seed(args.seed + i))
                               for i in idxs])
            imgs.append(model.sample([tp] * len(idxs), noise=noise, guidance_scale=args.cfg,
                                     num_steps=args.steps).cpu())
        imgs = torch.cat(imgs)
        gen_by_p[tp] = imgs
        pred = predict_passage(cls, imgs, device)
        gm = imgs.mean(dim=(1, 2, 3)).numpy()
        bright[tp] = (float(gm.mean()), float(imgs.std(dim=(1, 2, 3)).mean()))
        grid_imgs[tp] = imgs[:4]
        rows_a.append({'target_passage': tp, 'pred_mean': float(pred.mean()), 'pred_std': float(pred.std()),
                       'pred_median': float(np.median(pred)), 'mean_intensity': bright[tp][0],
                       'mean_local_std': bright[tp][1]})
        print(f"    P{tp:>2}: predicted {pred.mean():5.1f} ± {pred.std():4.1f} | intensity {bright[tp][0]:.3f}")
    df_a = pd.DataFrame(rows_a)
    # paired-noise Spearman over all samples
    all_t = np.repeat(targets, n)
    all_p = np.concatenate([predict_passage(cls, gen_by_p[t], device) for t in targets])
    rho_all = float(spearmanr(all_t, all_p).correlation)
    ms_a = monotone_stats(df_a['target_passage'], df_a['pred_mean'])
    print(f"    -> Spearman(all samples)={rho_all:.3f} | Spearman(means)={ms_a['spearman_means']:.3f} | "
          f"adjacent-increasing={ms_a['frac_adjacent_increasing']:.2f} | slope={ms_a['slope']:.2f} (ideal 1.0)")

    # montage: rows = target passages, cols = 4 shared noises
    save_image(torch.cat([grid_imgs[t] for t in targets]), os.path.join(out, 'montage_noise2img.png'), nrow=4, padding=2)

    # ---------------------------------------------------------------- Part B: SDEdit
    print("\n[B] SDEdit translation of REAL held-out tiles")
    rows_b = []
    def sdedit_block(name, src_df, tgts):
        src_df = src_df.iloc[np.linspace(0, len(src_df) - 1, min(args.n_src, len(src_df))).astype(int)]
        src = torch.stack([load_tile(f) for f in src_df['tile_filename']])
        src_pred = predict_passage(cls, src, device)
        print(f"    {name}: {len(src)} source tiles | source true P{src_df['passage'].mean():.1f} "
              f"| source predicted {src_pred.mean():.1f}")
        mean_by_t = {}
        for tp in tgts:
            outs = []
            for s in range(0, len(src), args.batch):
                x = (src[s:s + args.batch].to(device) * 2 - 1)
                g = torch.Generator(device=device).manual_seed(args.seed + s)
                noise = torch.randn(x.shape[0], 4, 64, 64, device=device, generator=g)
                outs.append(model.translate(x, tp, strength=args.strength, guidance_scale=args.cfg,
                                            num_steps=args.steps, noise=noise).cpu())
            outs = torch.cat(outs)
            pr = predict_passage(cls, outs, device)
            mean_by_t[tp] = float(pr.mean())
            rows_b.append({'direction': name, 'target_passage': tp, 'src_true_mean': float(src_df['passage'].mean()),
                           'src_pred_mean': float(src_pred.mean()), 'pred_mean': float(pr.mean()), 'pred_std': float(pr.std())})
            print(f"      -> target P{tp:>2}: predicted {pr.mean():5.1f} ± {pr.std():4.1f}")
            if tp == tgts[-1] or tp == tgts[0]:
                save_image(torch.cat([src[:4], outs[:4]]), os.path.join(out, f'sdedit_{name}_to_P{tp}.png'), nrow=4)
        return monotone_stats(list(mean_by_t.keys()), list(mean_by_t.values())), src_pred.mean()

    young_src = meta[meta['passage'] <= 8]
    old_src = meta[meta['passage'] >= 28]
    ms_aging, _ = sdedit_block('aging', young_src, [10, 15, 20, 25, 30, 35, 40])
    ms_rejuv, _ = sdedit_block('rejuvenation', old_src, [5, 10, 15, 20, 25])
    df_b = pd.DataFrame(rows_b)
    print(f"    aging slope={ms_aging['slope']:.2f} (Spearman {ms_aging['spearman_means']:.2f}) | "
          f"rejuv slope={ms_rejuv['slope']:.2f} (Spearman {ms_rejuv['spearman_means']:.2f})")

    # ---------------------------------------------------------------- Part D: intensity shortcut
    rho_int = float(spearmanr(df_a['target_passage'], df_a['mean_intensity']).correlation)
    print(f"\n[D] Intensity shortcut: Spearman(target passage, mean intensity)={rho_int:+.2f} "
          f"(real hFB14 mean intensity {real_intensity:.3f})")

    # ---------------------------------------------------------------- Part C: FID
    fid_rows = []
    if not args.skip_fid:
        print("\n[C] FID per age group vs real hFB14 tiles of that group")
        groups = {'young(P<=8)': ([1, 5], lambda p: p <= 8),
                  'inter(P9-19)': ([10, 15], lambda p: (p > 8) & (p < 20)),
                  'senes(P>=20)': ([20, 25, 30, 35, 40], lambda p: p >= 20)}
        for gname, (tps, sel) in groups.items():
            safe = gname.split('(')[0]
            gdir = os.path.join(out, 'fid_tmp', f'gen_{safe}')
            rdir = os.path.join(out, 'fid_tmp', f'real_{safe}')
            os.makedirs(gdir, exist_ok=True); os.makedirs(rdir, exist_ok=True)
            k = 0
            for tp in tps:
                for im in gen_by_p[tp]:
                    save_image(im, os.path.join(gdir, f'g_{k:04d}.png')); k += 1
            real_rows = meta[sel(meta['passage'].to_numpy())]
            if len(os.listdir(rdir)) != len(real_rows):
                for f in real_rows['tile_filename']:
                    save_image(load_tile(f), os.path.join(rdir, os.path.splitext(f)[0] + '.png'))
            fid = stage_fid(gdir, rdir, device)
            fid_rows.append({'group': gname, 'n_gen': k, 'n_real': len(real_rows), 'FID': fid})
            print(f"    {gname}: FID={fid:.1f} (gen N={k}, real N={len(real_rows)})")
    df_c = pd.DataFrame(fid_rows)

    # ---------------------------------------------------------------- outputs
    df_a.to_csv(os.path.join(out, 'partA_noise2img_sweep.csv'), index=False)
    df_b.to_csv(os.path.join(out, 'partB_sdedit.csv'), index=False)
    if len(df_c):
        df_c.to_csv(os.path.join(out, 'partC_fid.csv'), index=False)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].plot([0, 42], [0, 42], 'k--', label='ideal (y=x)')
    cp = sorted(cal_means_by_p)
    ax[0].plot(cp, [cal_means_by_p[p] for p in cp], 'g-o', ms=3, label='REAL hFB14 (judge ceiling)')
    ax[0].errorbar(df_a['target_passage'], df_a['pred_mean'], yerr=df_a['pred_std'], fmt='r-s', capsize=3, label='v16 generated')
    ax[0].set_xlabel('conditioning passage'); ax[0].set_ylabel('classifier-predicted passage')
    ax[0].set_title('A. Noise→image sweep'); ax[0].legend(); ax[0].grid(alpha=.3)
    for d, c in (('aging', 'tab:blue'), ('rejuvenation', 'tab:orange')):
        s = df_b[df_b['direction'] == d]
        ax[1].errorbar(s['target_passage'], s['pred_mean'], yerr=s['pred_std'], fmt='-o', color=c, capsize=3, label=d)
    ax[1].plot([0, 42], [0, 42], 'k--'); ax[1].set_xlabel('target passage'); ax[1].set_ylabel('predicted passage')
    ax[1].set_title(f'B. SDEdit (strength {args.strength})'); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(df_a['target_passage'], df_a['mean_intensity'], 'm-o', label='generated mean intensity')
    ax[2].axhline(real_intensity, color='g', ls='--', label='real hFB14 mean')
    ax[2].set_xlabel('conditioning passage'); ax[2].set_title(f'D. Intensity shortcut (ρ={rho_int:+.2f})')
    ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'v16_evaluation_summary.png'), dpi=150); plt.close()

    def verdict(rho, slope, adj):
        if rho > 0.9 and adj >= 0.85 and 0.6 <= slope <= 1.4:
            return 'STRONG: passage control is monotone and well calibrated'
        if rho > 0.8 and adj >= 0.7:
            return 'OK: monotone but compressed/biased -> check slope & mid-passages'
        return 'WEAK: passage conditioning is not reliably controlling morphology'

    md = [f"# LDM v16 Stage-1 Evaluation ({tag})", "",
          f"Judge: ConvNeXt classifier v3 (never saw hFB14). Held-out donor: **{VAL_DONOR}**. "
          f"Sampling: {args.steps} DDIM steps, CFG {args.cfg}, seed {args.seed}, EMA weights.", "",
          "## 0. Judge ceiling on REAL held-out tiles",
          f"- Spearman(true, predicted passage) = **{real_rho:.3f}**, MAE = **{real_mae:.2f}** passages (N={len(cal)}).",
          "- Generated results cannot be expected to beat this ceiling.", "",
          "## A. Paired noise→image sweep", md_table(df_a), "",
          f"- Spearman (all samples) = **{rho_all:.3f}**; Spearman (means) = {ms_a['spearman_means']:.3f}; "
          f"adjacent-increasing = {ms_a['frac_adjacent_increasing']:.2f}; slope = **{ms_a['slope']:.2f}** (ideal 1.0).",
          f"- Verdict: **{verdict(ms_a['spearman_means'], ms_a['slope'], ms_a['frac_adjacent_increasing'])}**", "",
          "## B. SDEdit on real held-out tiles", md_table(df_b.round(2)), "",
          f"- Aging slope = {ms_aging['slope']:.2f} (Spearman {ms_aging['spearman_means']:.2f}); "
          f"Rejuvenation slope = {ms_rejuv['slope']:.2f} (Spearman {ms_rejuv['spearman_means']:.2f}).", ""]
    if len(df_c):
        md += ["## C. FID per age group (small generated N -> biased upward, compare groups not absolute values)",
               md_table(df_c.round(2)), ""]
    md += ["## D. Intensity shortcut check",
           f"- Spearman(target passage, generated mean intensity) = **{rho_int:+.2f}**. "
           "A value near ±1 together with weak classifier monotonicity would suggest the model mostly encodes "
           "brightness/batch rather than morphology.",
           "- Real hFB14 mean intensity by group: " + ', '.join(f'{k}={v:.3f}' for k, v in real_int_by_group.items())
           + f"; generated (mean over passages) = {float(df_a['mean_intensity'].mean()):.3f}. "
           "A large absolute gap means generated tiles are globally darker/lower-contrast than real ones "
           "(check CFG scale and CLAHE tile statistics).", "",
           f"Runtime: {(time.time() - t_start) / 60:.1f} min. Figures: `v16_evaluation_summary.png`, `montage_noise2img.png` "
           "(rows = P1,5,10,...,40; columns = 4 shared noises)."]
    with open(os.path.join(out, 'v16_evaluation_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"\n[DONE] {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == '__main__':
    main()
