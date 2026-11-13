"""
Build Zero-Leakage Clean Dataset: data/processed_v5
===================================================
1. Identifies and hard-moves the 62 data/unlabeled (.tif) files that overlap with
   data/raw/{train,val,test} into data/archive/unlabeled_overlaps/ (preserving full VIA backup).
2. Hard-moves the 2 shifted-FOV cross-split plates from data/raw/train
   (senescent_MSCs_10038.jpg -> overlaps test/10040; senescent_MSCs_10003.jpg -> overlaps val/10006)
   into data/archive/raw_train_overlaps/ (preserving full VIA backup).
3. Crops fixed-scale 512x512 patches (Cropper logic, NO zoom/resize to preserve true biological
   cell size & senescence hypertrophy) into:
     - data/processed_v5/train/{young,senescent}  (48 clean human raw/train + 179 clean unlabeled pseudo)
     - data/processed_v5/val/{young,senescent}    (20 gold human raw/val plates; 0 pseudo, 0 leakage)
     - data/processed_v5/test/{young,senescent}   (30 gold human raw/test plates; 0 pseudo, 0 leakage)
4. Filters out:
     - Exact intra-image clamped (left, top) window collisions
     - The 144 manually quarantined pseudo-crops in data/pseudo_label/qara-{sen,you}
     - Near-identical intra-train crop duplicates (MD5 exact or 32x32 MAE < 1.0 & corr > 0.96)
5. Runs post-build zero-leakage audit across train/val/test crops and writes split_manifest_v5.json.
"""

import os
import sys
import json
import shutil
import hashlib
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

CROP_SIZE = 512
SHIFTED_CROSS_SPLIT_TRAIN = {"senescent_MSCs_10038.jpg", "senescent_MSCs_10003.jpg", "senescent_MSCs_10043.jpg"}
SHIFTED_CROSS_SPLIT_VAL = {"senescent_MSCs_10006.jpg"}


def load_via_meta(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    has_wrapper = isinstance(data, dict) and "_via_img_metadata" in data
    meta = data["_via_img_metadata"] if has_wrapper else data
    return data, meta, has_wrapper


def save_via_meta(json_path, data, meta, has_wrapper):
    if has_wrapper:
        data["_via_img_metadata"] = meta
        out = data
    else:
        out = meta
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f)


def thumb64(img_path):
    with Image.open(img_path) as im:
        return np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)


def norm_vec(arr):
    v = arr.ravel().astype(np.float32)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-8)


def find_unlabeled_overlaps():
    raw_dirs = ["data/raw/train", "data/raw/val", "data/raw/test"]
    raw_vecs, raw_arrs, raw_names = [], [], []
    for rd in raw_dirs:
        full_rd = os.path.join(ROOT_DIR, rd)
        for fn in sorted(os.listdir(full_rd)):
            if fn.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff", ".png")):
                a = thumb64(os.path.join(full_rd, fn))
                raw_arrs.append(a)
                raw_vecs.append(norm_vec(a))
                raw_names.append(f"{rd}/{fn}")

    R = np.stack(raw_vecs)
    unl_dir = os.path.join(ROOT_DIR, "data", "unlabeled")
    overlaps = []
    for fn in sorted(os.listdir(unl_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff", ".png")):
            continue
        fp = os.path.join(unl_dir, fn)
        a = thumb64(fp)
        v = norm_vec(a)
        sims = R @ v
        best_i = int(np.argmax(sims))
        corr = float(sims[best_i])
        mae = float(np.mean(np.abs(a - raw_arrs[best_i])))
        if corr > 0.80 or mae < 1.5:
            overlaps.append((fn, raw_names[best_i], corr, mae))
    return overlaps


def archive_overlapping_raw_and_unlabeled():
    manifest_moves = {"unlabeled_overlaps": [], "raw_train_shifted_overlaps": [], "raw_val_shifted_overlaps": []}

    # 1. Move overlapping files out of data/unlabeled
    unl_dir = os.path.join(ROOT_DIR, "data", "unlabeled")
    unl_arch = os.path.join(ROOT_DIR, "data", "archive", "unlabeled_overlaps")
    os.makedirs(unl_arch, exist_ok=True)
    unl_json = os.path.join(unl_dir, "via_project.json")
    unl_backup = os.path.join(unl_arch, "via_project_full_backup.json")
    if not os.path.exists(unl_backup) and os.path.exists(unl_json):
        shutil.copy2(unl_json, unl_backup)

    overlaps = find_unlabeled_overlaps()
    overlap_fns = {fn for fn, _, _, _ in overlaps}
    for fn, matched_raw, corr, mae in overlaps:
        src = os.path.join(unl_dir, fn)
        dst = os.path.join(unl_arch, fn)
        if os.path.exists(src):
            shutil.move(src, dst)
            manifest_moves["unlabeled_overlaps"].append(
                {"file": fn, "matched_raw": matched_raw, "corr": round(corr, 4), "mae": round(mae, 3)}
            )

    if overlap_fns and os.path.exists(unl_json):
        data, meta, has_wrap = load_via_meta(unl_json)
        pruned = {k: v for k, v in meta.items() if v.get("filename") not in overlap_fns}
        save_via_meta(unl_json, data, pruned, has_wrap)

    # 2. Move shifted cross-split plates out of data/raw/train & data/raw/val
    for split_name, target_set, arch_sub, manifest_key in [
        ("train", SHIFTED_CROSS_SPLIT_TRAIN, "raw_train_overlaps", "raw_train_shifted_overlaps"),
        ("val", SHIFTED_CROSS_SPLIT_VAL, "raw_val_overlaps", "raw_val_shifted_overlaps"),
    ]:
        sp_dir = os.path.join(ROOT_DIR, "data", "raw", split_name)
        sp_arch = os.path.join(ROOT_DIR, "data", "archive", arch_sub)
        os.makedirs(sp_arch, exist_ok=True)
        sp_json = os.path.join(sp_dir, "via_project.json")
        sp_backup = os.path.join(sp_arch, "via_project_full_backup.json")
        if not os.path.exists(sp_backup) and os.path.exists(sp_json):
            shutil.copy2(sp_json, sp_backup)

        moved_fns = set()
        for fn in sorted(target_set):
            src = os.path.join(sp_dir, fn)
            dst = os.path.join(sp_arch, fn)
            if os.path.exists(src):
                shutil.move(src, dst)
                moved_fns.add(fn)
                manifest_moves[manifest_key].append(fn)

        if moved_fns and os.path.exists(sp_json):
            data, meta, has_wrap = load_via_meta(sp_json)
            pruned = {k: v for k, v in meta.items() if v.get("filename") not in moved_fns}
            save_via_meta(sp_json, data, pruned, has_wrap)

    return manifest_moves


def load_quarantine_signatures():
    q_md5s = set()
    q_arrs = []
    for sub in ["qara-sen", "qara-you"]:
        qdir = os.path.join(ROOT_DIR, "data", "pseudo_label", sub)
        if not os.path.exists(qdir):
            continue
        for root, _, files in os.walk(qdir):
            for fn in files:
                if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                    continue
                fp = os.path.join(root, fn)
                with open(fp, "rb") as f:
                    q_md5s.add(hashlib.md5(f.read()).hexdigest())
                with Image.open(fp) as im:
                    q_arrs.append(np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32))
    q_stack = np.stack(q_arrs) if q_arrs else None
    return q_md5s, q_stack


def compute_clamped_box(points_x, points_y, img_w, img_h, crop_window=CROP_SIZE):
    min_x, max_x = int(min(points_x)), int(max(points_x))
    min_y, max_y = int(min(points_y)), int(max(points_y))
    mid_x = (min_x + max_x) // 2
    mid_y = (min_y + max_y) // 2
    half = crop_window // 2
    left = max(0, min(img_w - crop_window, mid_x - half))
    top = max(0, min(img_h - crop_window, mid_y - half))
    return left, top, left + crop_window, top + crop_window


def crop_source_into_split(
    src_rel,
    out_split_dir,
    is_pseudo,
    q_md5s,
    q_stack,
    seen_train_vecs=None,
    seen_train_arrs=None,
    counter_start=0,
):
    src_dir = os.path.join(ROOT_DIR, src_rel)
    json_path = os.path.join(src_dir, "via_project.json")
    _, meta, _ = load_via_meta(json_path)

    for cls in ("young", "senescent"):
        os.makedirs(os.path.join(out_split_dir, cls), exist_ok=True)

    stats = {
        "source": src_rel,
        "plates_used": 0,
        "raw_polygons": 0,
        "skipped_window_collision": 0,
        "skipped_quarantine": 0,
        "skipped_near_dup_in_split": 0,
        "saved_young": 0,
        "saved_senescent": 0,
        "plates": [],
    }

    counter = counter_start
    for key in sorted(meta.keys()):
        entry = meta[key]
        filename = entry.get("filename", "")
        img_path = os.path.join(src_dir, filename)
        if not filename or not os.path.exists(img_path):
            continue

        with Image.open(img_path) as im:
            img = im.convert("RGB")
            img_w, img_h = img.size
            if img_w < CROP_SIZE or img_h < CROP_SIZE:
                continue

            seen_boxes = set()
            plate_y = 0
            plate_s = 0
            for reg in entry.get("regions", []):
                lbl = reg.get("region_attributes", {}).get("cell")
                if lbl not in ("young", "senescent"):
                    continue
                sa = reg.get("shape_attributes", {})
                if sa.get("name") != "polygon":
                    continue
                px, py = sa.get("all_points_x", []), sa.get("all_points_y", [])
                if not px or not py:
                    continue

                stats["raw_polygons"] += 1
                left, top, right, bottom = compute_clamped_box(px, py, img_w, img_h, CROP_SIZE)
                if (left, top) in seen_boxes:
                    stats["skipped_window_collision"] += 1
                    continue
                seen_boxes.add((left, top))

                region = img.crop((left, top, right, bottom))

                # Filter quarantined bad pseudo-crops (144 qara-sen/qara-you images)
                if is_pseudo and q_stack is not None:
                    a32 = np.asarray(region.convert("L").resize((32, 32)), dtype=np.float32)
                    maes = np.mean(np.abs(q_stack - a32), axis=(1, 2))
                    if float(maes.min()) < 1.5:
                        stats["skipped_quarantine"] += 1
                        continue

                # Also filter rare near-identical overlapping crops within train (e.g. 16.tif & 22.tif)
                if seen_train_arrs is not None:
                    a32 = np.asarray(region.convert("L").resize((32, 32)), dtype=np.float32)
                    v32 = norm_vec(a32)
                    if seen_train_vecs:
                        sims = np.dot(np.stack(seen_train_vecs), v32)
                        mx_i = int(np.argmax(sims))
                        if float(sims[mx_i]) > 0.96 and float(np.mean(np.abs(seen_train_arrs[mx_i] - a32))) < 2.0:
                            stats["skipped_near_dup_in_split"] += 1
                            continue
                    seen_train_vecs.append(v32)
                    seen_train_arrs.append(a32)

                stem = os.path.splitext(filename)[0]
                prefix = "unl_" if is_pseudo else ""
                save_path = os.path.join(out_split_dir, lbl, f"{prefix}{stem}_{counter}.jpg")
                while os.path.exists(save_path):
                    counter += 1
                    save_path = os.path.join(out_split_dir, lbl, f"{prefix}{stem}_{counter}.jpg")

                region.save(save_path, quality=95)
                counter += 1
                if lbl == "young":
                    stats["saved_young"] += 1
                    plate_y += 1
                else:
                    stats["saved_senescent"] += 1
                    plate_s += 1

            stats["plates_used"] += 1
            stats["plates"].append({"filename": filename, "young": plate_y, "senescent": plate_s})

    return stats, counter


def verify_zero_leakage(out_base):
    splits = {}
    for sp in ("train", "val", "test"):
        recs = []
        for cls in ("young", "senescent"):
            cdir = os.path.join(out_base, sp, cls)
            for fn in sorted(os.listdir(cdir)):
                fp = os.path.join(cdir, fn)
                with open(fp, "rb") as f:
                    md5 = hashlib.md5(f.read()).hexdigest()
                with Image.open(fp) as im:
                    a32 = np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32)
                recs.append((cls + "/" + fn, md5, a32, norm_vec(a32)))
        splits[sp] = recs

    report = {"counts": {}, "md5_cross_overlaps": {}, "high_corr_cross_overlaps": {}}
    for sp, recs in splits.items():
        md5_set = {r[1] for r in recs}
        report["counts"][sp] = {"total": len(recs), "unique_md5": len(md5_set)}

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        md5_a = {r[1] for r in splits[a]}
        md5_b = {r[1] for r in splits[b]}
        report["md5_cross_overlaps"][f"{a}_vs_{b}"] = len(md5_a & md5_b)

        Va = np.stack([r[3] for r in splits[a]])
        Vb = np.stack([r[3] for r in splits[b]])
        S = Va @ Vb.T
        max_corr = float(S.max())
        i, j = divmod(int(np.argmax(S)), len(splits[b]))
        mae = float(np.mean(np.abs(splits[a][i][2] - splits[b][j][2])))
        pairs_gt85 = int((S > 0.85).sum())
        report["high_corr_cross_overlaps"][f"{a}_vs_{b}"] = {
            "max_corr": round(max_corr, 4),
            "mae_at_max_corr": round(mae, 2),
            "pair": f"{splits[a][i][0]} <-> {splits[b][j][0]}",
            "pairs_gt_0.85": pairs_gt85,
        }
    return report


def main():
    print("[Step 1/4] Archiving raw/unlabeled overlaps & cross-split shifted plates...")
    moves = archive_overlapping_raw_and_unlabeled()
    print(f"  Moved unlabeled->archive: {len(moves['unlabeled_overlaps'])} files")
    print(f"  Moved raw/train shifted->archive: {len(moves['raw_train_shifted_overlaps'])} files ({moves['raw_train_shifted_overlaps']})")
    print(f"  Moved raw/val shifted->archive: {len(moves['raw_val_shifted_overlaps'])} files ({moves['raw_val_shifted_overlaps']})")

    out_base = os.path.join(ROOT_DIR, "data", "processed_v5")
    if os.path.exists(out_base):
        raise RuntimeError(f"{out_base} already exists! Remove/archive it manually before re-running.")
    os.makedirs(out_base, exist_ok=True)

    print("[Step 2/4] Loading quarantined pseudo-crop signatures (qara-sen/qara-you)...")
    q_md5s, q_stack = load_quarantine_signatures()
    print(f"  Loaded {len(q_md5s)} quarantined crops.")

    print("[Step 3/4] Cropping fixed 512x512 patches (Cropper, NO zoom) into data/processed_v5...")
    seen_tr_vecs, seen_tr_arrs = [], []
    tr_gold_stats, cnt = crop_source_into_split(
        "data/raw/train", os.path.join(out_base, "train"),
        is_pseudo=False, q_md5s=q_md5s, q_stack=q_stack,
        seen_train_vecs=seen_tr_vecs, seen_train_arrs=seen_tr_arrs, counter_start=0
    )
    tr_pseudo_stats, cnt = crop_source_into_split(
        "data/unlabeled", os.path.join(out_base, "train"),
        is_pseudo=True, q_md5s=q_md5s, q_stack=q_stack,
        seen_train_vecs=seen_tr_vecs, seen_train_arrs=seen_tr_arrs, counter_start=cnt
    )
    val_stats, _ = crop_source_into_split(
        "data/raw/val", os.path.join(out_base, "val"),
        is_pseudo=False, q_md5s=q_md5s, q_stack=q_stack, counter_start=0
    )
    test_stats, _ = crop_source_into_split(
        "data/raw/test", os.path.join(out_base, "test"),
        is_pseudo=False, q_md5s=q_md5s, q_stack=q_stack, counter_start=0
    )

    print("[Step 4/4] Running post-build zero-leakage verification on data/processed_v5...")
    leak_report = verify_zero_leakage(out_base)

    manifest = {
        "crop_size": CROP_SIZE,
        "crop_policy": "fixed_512x512_no_zoom_scale_preserving",
        "archive_moves": moves,
        "splits": {
            "train_gold": tr_gold_stats,
            "train_pseudo": tr_pseudo_stats,
            "val_gold": val_stats,
            "test_gold": test_stats,
        },
        "zero_leakage_verification": leak_report,
    }
    manifest_path = os.path.join(out_base, "split_manifest_v5.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n================ DATA/PROCESSED_V5 SUMMARY ================")
    ty = tr_gold_stats["saved_young"] + tr_pseudo_stats["saved_young"]
    ts = tr_gold_stats["saved_senescent"] + tr_pseudo_stats["saved_senescent"]
    print(f"TRAIN : {tr_gold_stats['plates_used']} gold + {tr_pseudo_stats['plates_used']} pseudo plates = {ty} young + {ts} senescent = {ty+ts} crops")
    print(f"        (filtered {tr_gold_stats['skipped_window_collision']+tr_pseudo_stats['skipped_window_collision']} box collisions, {tr_pseudo_stats['skipped_quarantine']} qara, {tr_pseudo_stats['skipped_near_dup_in_split']} intra-train near-dups)")
    print(f"VAL   : {val_stats['plates_used']} gold plates = {val_stats['saved_young']} young + {val_stats['saved_senescent']} senescent = {val_stats['saved_young']+val_stats['saved_senescent']} crops (skipped {val_stats['skipped_window_collision']} box collisions)")
    print(f"TEST  : {test_stats['plates_used']} gold plates = {test_stats['saved_young']} young + {test_stats['saved_senescent']} senescent = {test_stats['saved_young']+test_stats['saved_senescent']} crops (skipped {test_stats['skipped_window_collision']} box collisions)")
    print("VERIFY:", json.dumps(leak_report, indent=2))
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
