# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label
from scipy import stats

d_in_young = np.load('scratch/mrcnn_masks_input_young.npy', allow_pickle=True).item()
d_out_aging = np.load('scratch/mrcnn_masks_aging.npy', allow_pickle=True).item()
d_cg_aging = np.load('scratch/mrcnn_masks_cyclegan_aging.npy', allow_pickle=True).item()

d_in_sen = np.load('scratch/mrcnn_masks_input_senescent.npy', allow_pickle=True).item()
d_out_reju = np.load('scratch/mrcnn_masks_reju.npy', allow_pickle=True).item()
d_cg_reju = np.load('scratch/mrcnn_masks_cyclegan_reju.npy', allow_pickle=True).item()

def get_props(mask):
    if np.sum(mask) == 0:
        return None
    lbl = label(mask)
    props = regionprops(lbl)
    if len(props) == 0:
        return None
    p = max(props, key=lambda x: x.area)
    if p.minor_axis_length == 0:
        return None
    return {
        'area': float(p.area),
        'eccentricity': float(p.eccentricity),
        'aspect_ratio': float(p.major_axis_length / p.minor_axis_length)
    }

def get_fov(fname):
    m = re.match(r'^(.*)_[0-9]+\.(?:jpg|png)$', fname)
    return m.group(1) if m else fname

# Aging pairs (LDM and CycleGAN)
aging_records = []
for fname in sorted(d_in_young.keys()):
    p_in = get_props(d_in_young[fname])
    p_ldm = get_props(d_out_aging[fname])
    p_cg = get_props(d_cg_aging[fname])
    
    if p_in is not None and p_ldm is not None:
        rec = {
            'fname': fname,
            'fov': get_fov(fname),
            'in_ar': p_in['aspect_ratio'],
            'ldm_ar': p_ldm['aspect_ratio'],
            'in_ecc': p_in['eccentricity'],
            'ldm_ecc': p_ldm['eccentricity'],
            'ldm_delta_ar': p_ldm['aspect_ratio'] - p_in['aspect_ratio'],
            'ldm_delta_ecc': p_ldm['eccentricity'] - p_in['eccentricity']
        }
        if p_cg is not None:
            rec['cg_ar'] = p_cg['aspect_ratio']
            rec['cg_ecc'] = p_cg['eccentricity']
            rec['cg_delta_ar'] = p_cg['aspect_ratio'] - p_in['aspect_ratio']
            rec['cg_delta_ecc'] = p_cg['eccentricity'] - p_in['eccentricity']
        aging_records.append(rec)

# Rejuv pairs
reju_records = []
for fname in sorted(d_in_sen.keys()):
    p_in = get_props(d_in_sen[fname])
    p_ldm = get_props(d_out_reju[fname])
    p_cg = get_props(d_cg_reju[fname])
    
    if p_in is not None and p_ldm is not None:
        rec = {
            'fname': fname,
            'fov': get_fov(fname),
            'in_ar': p_in['aspect_ratio'],
            'ldm_ar': p_ldm['aspect_ratio'],
            'in_ecc': p_in['eccentricity'],
            'ldm_ecc': p_ldm['eccentricity'],
            'ldm_delta_ar': p_ldm['aspect_ratio'] - p_in['aspect_ratio'],
            'ldm_delta_ecc': p_ldm['eccentricity'] - p_in['eccentricity']
        }
        if p_cg is not None:
            rec['cg_ar'] = p_cg['aspect_ratio']
            rec['cg_ecc'] = p_cg['eccentricity']
            rec['cg_delta_ar'] = p_cg['aspect_ratio'] - p_in['aspect_ratio']
            rec['cg_delta_ecc'] = p_cg['eccentricity'] - p_in['eccentricity']
        reju_records.append(rec)

df_a = pd.DataFrame(aging_records)
df_r = pd.DataFrame(reju_records)

print(f"Aging valid cell pairs: {len(df_a)}, across {df_a['fov'].nunique()} FOVs")
print(f"Reju valid cell pairs: {len(df_r)}, across {df_r['fov'].nunique()} FOVs")

# FOV aggregation
fov_a = df_a.groupby('fov').mean(numeric_only=True)
fov_r = df_r.groupby('fov').mean(numeric_only=True)

results = []

def analyze(name, series):
    diff = series.dropna()
    t_stat, p_val = stats.ttest_1samp(diff, 0)
    w_res = stats.wilcoxon(diff)
    ci = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
    res = {
        'comparison': name,
        'mean_diff': np.mean(diff),
        'ci_low': ci[0],
        'ci_high': ci[1],
        'df': len(diff) - 1,
        't_stat': t_stat,
        'p_val': p_val,
        'wilcoxon_p': w_res.pvalue
    }
    results.append(res)
    print(f"{name:30s}: Delta={np.mean(diff):+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], t({len(diff)-1})={t_stat:+.4f}, p={p_val:.4e}, wilcoxon_p={w_res.pvalue:.4e}")

print("\n--- FOV-LEVEL PAIRED ANALYSIS (8 COMPARISONS) ---")
analyze("Aging LDM Delta AR", fov_a['ldm_delta_ar'])
analyze("Aging LDM Delta Ecc", fov_a['ldm_delta_ecc'])
analyze("Reju LDM Delta AR", fov_r['ldm_delta_ar'])
analyze("Reju LDM Delta Ecc", fov_r['ldm_delta_ecc'])

analyze("Aging CG Delta AR", fov_a['cg_delta_ar'])
analyze("Aging CG Delta Ecc", fov_a['cg_delta_ecc'])
analyze("Reju CG Delta AR", fov_r['cg_delta_ar'])
analyze("Reju CG Delta Ecc", fov_r['cg_delta_ecc'])

# Holm correction across the 8 FOV p-values
p_vals = [r['p_val'] for r in results]
sorted_indices = np.argsort(p_vals)
m = len(p_vals)
adjusted_p = [0.0] * m

running_max = 0.0
for rank, idx in enumerate(sorted_indices):
    p = p_vals[idx]
    adj = (m - rank) * p
    running_max = max(running_max, adj)
    adjusted_p[idx] = min(running_max, 1.0)

for r, adj in zip(results, adjusted_p):
    r['holm_p'] = adj

print("\n--- HOLM-ADJUSTED P-VALUES ---")
for r in results:
    print(f"{r['comparison']:30s}: raw_p={r['p_val']:.4e} -> Holm_p={r['holm_p']:.4e} (Sig: {r['holm_p'] < 0.05})")
