"""
rock_revision.py  -  additional experiments requested in the IJIES review of
"Open-Set and Hierarchical Lithology Recognition from Rock Photographs" (paper 20266719).

Run the steps in order (the Colab notebook does this for you):

  audit      leakage audit, units, folds, D2 export counts, dHash pair sample   (CPU, minutes)
  features   handcrafted descriptor (all images) + frozen embeddings (units)     (GPU, ~10 min)
  leakage    Table 5 rows incl. the new source-grouped row                       (CPU, minutes)
  probe      collection probe on all units and on Basalt only                    (CPU, minutes)
  finetune   5-fold P1, O1/O2 main runs, P3b runs (fine-tuned)                   (GPU, 2-4 h)
  analyse    validation-based score selection, Holm, bootstrap, P3b distance     (CPU, minutes)
  dhash_eval after you label the sampled pairs (optional)                        (CPU, seconds)

Everything is written to CONFIG["OUT"]. Zip that folder and send it back.
Every step can be re-run; finished fine-tuning runs are skipped (resume-safe).
"""
import argparse, hashlib, json, math, os, random, sys, time, glob, io, base64
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# =============================================================================
# CONFIG  (edit in the notebook through rock_revision.CONFIG.update({...}))
# =============================================================================
CONFIG = dict(
    D1_ROOT="/content/data/D1",        # contains <family>/<type>/*.jpg  (any depth; type = parent folder)
    D2_ROOT="/content/data/D2",        # Roboflow export: train/ valid/ test/ (folder or _classes.csv format)
    OUT="/content/drive/MyDrive/rock_revision_out",
    SEEDS=[0, 1, 2],
    NFOLD=5,
    DHASH_BITS_THRESHOLD=20,
    # Optional: CSV with columns md5,seed,fold to reuse the ORIGINAL folds of the paper (recommended).
    FOLDS_CSV=None,
    # Optional: python file defining describe(pil_image)->1-D np.ndarray (your ORIGINAL 178-d descriptor).
    DESCRIPTOR_PY=None,
    # timm names: put EXACTLY the names used in the original notebook.
    BACKBONES={
        "ResNet-50": "resnet50",
        "EfficientNet-B0": "efficientnet_b0",
        "ConvNeXt-T": "convnext_tiny",
        "ViT-S/16": "vit_small_patch16_224",
    },
    # Recent lightweight comparator (2024) trained under the same budget (reviewer 2, comment 1). None to skip.
    COMPARATOR={"MobileNetV4-M": "mobilenetv4_conv_medium"},
    PRETRAINED=True,
    # fine-tuning (Table 3 of the paper)
    IMG_SIZE=224, EPOCHS=10, WARMUP_EPOCHS=1, BATCH=64, LR_BACKBONE=1e-4, LR_HEAD=1e-3, WD=0.05,
    FAMILY_LAMBDA=0.5, CROP_SCALE=(0.5, 1.0), FLIP_P=0.5, VAL_FRACTION=0.10, NUM_WORKERS=4,
    # which fine-tuning jobs to run
    FT5_MODELS=[("ConvNeXt-T", False), ("ConvNeXt-T", True), ("ViT-S/16", False), ("ViT-S/16", True)],
    FT5_SEED=0,
    OSR_BACKBONES=["ResNet-50", "EfficientNet-B0", "ConvNeXt-T", "ViT-S/16"],
    P3B_SEEDS=[0],
    # P3b held-out types. REPLACE with the exact splits used in the paper (Table 8).
    P3B_SPLITS={
        "D1": [["Granite", "Marble"], ["Quartzite", "Coal"], ["Limestone", "Sandstone"]],
        "D2": [["Clay", "Chert", "Gypsum"], ["Conglomerate", "Diatomite", "Olivine basalt"],
               ["Shale (mudstone)", "Siliceous sinter", "Basalt"]],
    },
    # classical heads (put the ORIGINAL fixed hyper-parameters here)
    LR_C=1.0, SVM_C=10.0, SVM_GAMMA="scale",
    N_BOOT=2000,
    DHASH_PAIRS_PER_BIN=25,
    FAST_DEBUG=False,
)

# ----------------------------------------------------------------------------- taxonomy
TAXONOMY = {  # type: (family, subgroup)
    "Granite": ("Igneous", "Plutonic"), "Basalt": ("Igneous", "Volcanic"),
    "Olivine basalt": ("Igneous", "Volcanic"),
    "Quartzite": ("Metamorphic", "Non-foliated"), "Marble": ("Metamorphic", "Non-foliated"),
    "Sandstone": ("Sedimentary", "Siliciclastic"), "Conglomerate": ("Sedimentary", "Siliciclastic"),
    "Shale (mudstone)": ("Sedimentary", "Siliciclastic"), "Clay": ("Sedimentary", "Siliciclastic"),
    "Limestone": ("Sedimentary", "Chemical/biochemical"), "Chert": ("Sedimentary", "Chemical/biochemical"),
    "Diatomite": ("Sedimentary", "Chemical/biochemical"), "Siliceous sinter": ("Sedimentary", "Chemical/biochemical"),
    "Gypsum": ("Sedimentary", "Chemical/biochemical"), "Coal": ("Sedimentary", "Organic"),
}
TYPES = list(TAXONOMY)
FAMILIES = ["Igneous", "Metamorphic", "Sedimentary"]

def norm_label(name):
    k = name.lower().replace("_", " ").replace("-", " ").strip()
    k = " ".join(k.split())
    alias = {"olivine basalt": "Olivine basalt", "shale (mudstone)": "Shale (mudstone)",
             "shale mudstone": "Shale (mudstone)", "shale(mudstone)": "Shale (mudstone)", "shale": "Shale (mudstone)",
             "mudstone": "Shale (mudstone)", "siliceous sinter": "Siliceous sinter", "sinter": "Siliceous sinter"}
    if k in alias: return alias[k]
    for t in TYPES:
        if t.lower() == k: return t
    return None

def tax_distance(a, b):
    if a == b: return 0
    if TAXONOMY[a][1] == TAXONOMY[b][1] and TAXONOMY[a][0] == TAXONOMY[b][0]: return 1
    if TAXONOMY[a][0] == TAXONOMY[b][0]: return 2
    return 3

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}

def out(*p):
    d = os.path.join(CONFIG["OUT"], *p[:-1]) if len(p) > 1 else CONFIG["OUT"]
    os.makedirs(d, exist_ok=True)
    return os.path.join(CONFIG["OUT"], *p)

def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(out("log.txt"), "a") as f: f.write(msg + "\n")

# =============================================================================
# 1. AUDIT
# =============================================================================
def list_images():
    rows = []
    for dp, _, fs in os.walk(CONFIG["D1_ROOT"]):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                t = norm_label(os.path.basename(dp))
                rows.append(dict(path=os.path.join(dp, f), collection="D1", split="", raw=os.path.basename(dp), type=t, stem=""))
    root = CONFIG["D2_ROOT"]
    SPLITS = {"train", "valid", "validation", "val", "test"}
    subs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not any(d.lower() in SPLITS for d in subs):
        # export without split folders (e.g. a Kaggle mirror): treat the root as one split
        log("D2: no train/valid/test folders found; split counts will be reported as 'all'")
        subs = ["."]
    for split in subs:
        if split != "." and split.lower() not in SPLITS: continue
        sd = os.path.normpath(os.path.join(root, split))
        split = "all" if split == "." else split
        if not os.path.isdir(sd): continue
        csvp = os.path.join(sd, "_classes.csv")
        if os.path.exists(csvp):  # Roboflow multi-class CSV export
            df = pd.read_csv(csvp); df.columns = [c.strip() for c in df.columns]
            for _, r in df.iterrows():
                cls = [c for c in df.columns[1:] if r[c] == 1]
                raw = cls[0] if len(cls) == 1 else "|".join(cls)
                fn = r[df.columns[0]]
                rows.append(dict(path=os.path.join(sd, fn), collection="D2", split=split, raw=raw,
                                 type=norm_label(raw) if len(cls) == 1 else None, stem=fn.split(".rf.")[0]))
        else:
            for dp, _, fs in os.walk(sd):
                for f in fs:
                    if os.path.splitext(f)[1].lower() in IMG_EXT:
                        raw = os.path.basename(dp)
                        rows.append(dict(path=os.path.join(dp, f), collection="D2", split=split, raw=raw,
                                         type=norm_label(raw), stem=f.split(".rf.")[0]))
    df = pd.DataFrame(rows)
    bad = df[df.type.isna()]
    if len(bad):
        print("Unmapped labels (edit norm_label aliases):", bad.groupby(["collection", "raw"]).size())
        raise SystemExit(1)
    return df.reset_index(drop=True)

def dhash256(path):
    im = Image.open(path); im = ImageOps.exif_transpose(im).convert("L").resize((17, 16), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits)  # 32 bytes

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def hamming_pairs(H, maxd):
    """All pairs (i<j) with Hamming distance <= maxd. H: (n,32) uint8."""
    n = len(H); res = []
    for s in range(0, n, 256):
        X = H[s:s + 256]
        D = POP[np.bitwise_xor(X[:, None, :], H[None, :, :])].sum(-1, dtype=np.int32)
        ii, jj = np.nonzero(D <= maxd)
        for i, j in zip(ii, jj):
            i2 = i + s
            if j > i2: res.append((i2, j, int(D[i, j])))
    return res

class DSU:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)

def step_audit():
    df = list_images(); log("images:", len(df), dict(df.groupby("collection").size()))
    md5, dh = [], []
    for p in df.path:
        with open(p, "rb") as f: md5.append(hashlib.md5(f.read()).hexdigest())
        dh.append(dhash256(p))
    df["md5"] = md5; H = np.stack(dh); df["dhash"] = [h.tobytes().hex() for h in H]
    T = CONFIG["DHASH_BITS_THRESHOLD"]
    pairs40 = hamming_pairs(H, 40)
    n = len(df); d = DSU(n)
    n_dh = 0
    for i, j, dist in pairs40:
        if dist <= T: d.u(i, j); n_dh += 1
    by_md5 = defaultdict(list)
    for i, m in enumerate(df.md5): by_md5[m].append(i)
    for v in by_md5.values():
        for k in v[1:]: d.u(v[0], k)
    by_stem = defaultdict(list)
    for i, r in df[df.collection == "D2"].iterrows(): by_stem[(r.stem, r.type)].append(i)
    for v in by_stem.values():
        for k in v[1:]: d.u(v[0], k)
    df["group"] = [d.f(i) for i in range(n)]
    gtypes = df.groupby("group").type.nunique()
    df["conflict"] = df.group.map(gtypes) > 1
    keep = df[~df.conflict].sort_values("md5").groupby("group").head(1).index
    df["is_unit"] = False; df.loc[keep, "is_unit"] = True
    df.to_csv(out("audit", "image_to_group.csv"), index=False)
    conf = df[df.conflict].sort_values("group")
    conf.to_csv(out("audit", "conflict_groups.csv"), index=False)
    units = df[df.is_unit].copy()
    units[["md5", "path", "collection", "type", "group"]].to_csv(out("audit", "units_manifest.csv"), index=False)

    # folds
    if CONFIG["FOLDS_CSV"]:
        fo = pd.read_csv(CONFIG["FOLDS_CSV"])
        folds = units[["md5"]].merge(fo, on="md5", how="left")
        if folds.fold.isna().any(): log("WARNING: some units missing from FOLDS_CSV:", folds.fold.isna().sum())
    else:
        from sklearn.model_selection import StratifiedKFold
        fl = []
        for s in CONFIG["SEEDS"]:
            skf = StratifiedKFold(CONFIG["NFOLD"], shuffle=True, random_state=s)
            lab = (units.collection + "|" + units.type).values
            for f, (_, te) in enumerate(skf.split(np.zeros(len(units)), lab)):
                for m in units.md5.values[te]: fl.append(dict(md5=m, seed=s, fold=f))
        folds = pd.DataFrame(fl)
    folds.to_csv(out("audit", "folds.csv"), index=False)

    # D2 export structure (reviewer 1, comment 4)
    d2 = df[df.collection == "D2"]
    st = d2.groupby("split").agg(images=("path", "size"), source_photos=("stem", "nunique")).reset_index()
    mult = d2.groupby(["split", "stem"]).size().reset_index(name="copies").groupby("split").copies.agg(
        lambda s: dict(Counter(s)))
    st["copies_per_source"] = st.split.map(mult)
    st["units_kept"] = st.split.map(units[units.collection == "D2"].groupby("split").size()).fillna(0).astype(int)
    st.to_csv(out("audit", "d2_export_counts.csv"), index=False)

    # chaining check: largest components and components whose members are far apart
    comp = []
    for g, idx in df.groupby("group").groups.items():
        idx = list(idx)
        if len(idx) < 2: continue
        hh = H[idx]
        D = POP[np.bitwise_xor(hh[:, None, :], hh[None, :, :])].sum(-1)
        comp.append(dict(group=g, size=len(idx), max_internal_hamming=int(D.max()),
                         types="|".join(sorted(set(df.type[idx]))), collections="|".join(sorted(set(df.collection[idx])))))
    comp = pd.DataFrame(comp).sort_values("size", ascending=False)
    comp.to_csv(out("audit", "components.csv"), index=False)

    summ = dict(
        images=int(n), images_D1=int((df.collection == "D1").sum()), images_D2=int((df.collection == "D2").sum()),
        dhash_pairs_le_threshold=int(n_dh), md5_duplicate_groups=int(sum(len(v) > 1 for v in by_md5.values())),
        md5_duplicate_images=int(sum(len(v) for v in by_md5.values() if len(v) > 1)),
        groups=int(df.group.nunique()), conflict_groups=int(conf.group.nunique()), conflict_images=int(len(conf)),
        units=int(len(units)), units_D1=int((units.collection == "D1").sum()), units_D2=int((units.collection == "D2").sum()),
        largest_component=int(comp["size"].max()) if len(comp) else 1,
        components_with_internal_hamming_gt_40=int((comp.max_internal_hamming > 40).sum()) if len(comp) else 0,
        expected_from_paper=dict(units=3521, units_D1=1955, units_D2=1566, conflict_groups=23, conflict_images=79),
    )
    json.dump(summ, open(out("audit", "audit_summary.json"), "w"), indent=2)
    log("AUDIT", json.dumps(summ))
    if summ["units"] != 3521:
        log("WARNING: unit count differs from the paper (3,521). Check dataset versions and the hashing rule before going on.")

    # dHash validation sample: pairs across distance bins, exported for manual labelling
    rng = random.Random(0)
    bins = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40)]
    sel = []
    for lo, hi in bins:
        cand = [p for p in pairs40 if lo <= p[2] <= hi and df.md5[p[0]] != df.md5[p[1]]]
        rng.shuffle(cand); sel += cand[:CONFIG["DHASH_PAIRS_PER_BIN"]]
    rows, html = [], ["<html><body><h3>Label each pair: same photograph (incl. crop/flip/edit) = 1, different = 0</h3><table border=1>"]
    for k, (i, j, dist) in enumerate(sel):
        rows.append(dict(pair_id=k, path_a=df.path[i], path_b=df.path[j], hamming=dist,
                         same_stem=bool(df.stem[i] and df.stem[i] == df.stem[j]), same_photo=""))
        html.append("<tr><td>%d<br>d=%d</td><td>%s</td><td>%s</td></tr>" % (k, dist, _thumb(df.path[i]), _thumb(df.path[j])))
    html.append("</table></body></html>")
    pd.DataFrame(rows).to_csv(out("audit", "dhash_pairs_to_label.csv"), index=False)
    open(out("audit", "dhash_pairs.html"), "w").write("\n".join(html))

    # Basalt contact sheets for the visual inspection (reviewer 2, comment 3)
    for c in ["D1", "D2"]:
        b = units[(units.collection == c) & (units.type == "Basalt")]
        _sheet(list(b.path.sample(min(30, len(b)), random_state=0)), out("audit", "basalt_%s_sheet.jpg" % c))

def _thumb(p, s=160):
    im = Image.open(p).convert("RGB"); im.thumbnail((s, s)); b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    return '<img src="data:image/jpeg;base64,%s">' % base64.b64encode(b.getvalue()).decode()

def _sheet(paths, dst, s=200, cols=6):
    if not paths: return
    rows = math.ceil(len(paths) / cols); S = Image.new("RGB", (cols * s, rows * s), "white")
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB"); im = ImageOps.fit(im, (s - 4, s - 4))
        S.paste(im, ((k % cols) * s + 2, (k // cols) * s + 2))
    S.save(dst, quality=85)

def load_audit():
    df = pd.read_csv(out("audit", "image_to_group.csv"))
    folds = pd.read_csv(out("audit", "folds.csv"))
    units = df[df.is_unit].reset_index(drop=True)
    return df, units, folds

# =============================================================================
# 2. FEATURES
# =============================================================================
def _describe_default(im):
    """178-d colour-texture descriptor used ONLY if DESCRIPTOR_PY is not given.
    hsv hist 3x8=24 | rgb moments 9 | LBP-riu2 (8,1),(16,2),(24,3) = 10+18+26 = 54 |
    GLCM 6 props x 3 distances (4 angles averaged, 32 levels) = 18 |
    Gabor 4 freqs x 6 orient. mean+std = 48 | gradient: magnitude hist 16 + mean,std + orientation hist 7 = 25"""
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    from skimage.filters import gabor_kernel
    from scipy.signal import fftconvolve
    from scipy import ndimage
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size; s = 256 / min(w, h); im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = im.size; l, t = (w - 256) // 2, (h - 256) // 2; im = im.crop((l, t, l + 256, t + 256))
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    f = []
    for c in range(3): f += list(np.histogram(hsv[..., c], 8, (0, 1))[0] / hsv[..., c].size)
    for c in range(3):
        x = rgb[..., c].ravel(); m = x.mean(); sd = x.std() + 1e-8
        f += [m, sd, ((x - m) ** 3).mean() / sd ** 3]
    g = np.asarray(im.convert("L"), dtype=np.float32)
    for P, R in [(8, 1), (16, 2), (24, 3)]:
        lb = local_binary_pattern(g.astype(np.uint8), P, R, "uniform")
        f += list(np.histogram(lb, P + 2, (0, P + 2))[0] / lb.size)
    g32 = (g / 256 * 32).astype(np.uint8)
    M = graycomatrix(g32, [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 32, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        f += list(graycoprops(M, prop).mean(1))
    g128 = np.asarray(im.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
    g128 -= g128.mean()
    for fr in [0.05, 0.1, 0.2, 0.4]:
        for th in np.arange(6) * np.pi / 6:
            k = gabor_kernel(fr, theta=th)
            r = fftconvolve(g128, np.real(k), "same"); i_ = fftconvolve(g128, np.imag(k), "same")
            e = np.sqrt(r ** 2 + i_ ** 2); f += [e.mean(), e.std()]
    gx = ndimage.sobel(g / 255.0, 1); gy = ndimage.sobel(g / 255.0, 0); mag = np.hypot(gx, gy)
    f += list(np.histogram(mag, 16, (0, 4))[0] / mag.size) + [mag.mean(), mag.std()]
    ori = np.arctan2(gy, gx)
    f += list(np.histogram(ori, 7, (-np.pi, np.pi), weights=mag)[0] / (mag.sum() + 1e-8))
    v = np.asarray(f, dtype=np.float32); assert v.size == 178, v.size
    return v

def get_describe():
    if CONFIG["DESCRIPTOR_PY"]:
        import importlib.util
        sp = importlib.util.spec_from_file_location("userdesc", CONFIG["DESCRIPTOR_PY"]); m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m); log("using user descriptor from", CONFIG["DESCRIPTOR_PY"]); return m.describe
    log("using built-in 178-d descriptor (set DESCRIPTOR_PY to use the original one)")
    return _describe_default

def _desc_one(args):
    p, fn = args
    return fn(Image.open(p))

def step_features():
    df, units, _ = load_audit()
    fn = get_describe()
    dst = out("features", "handcrafted_all.npz")
    if not os.path.exists(dst):
        from joblib import Parallel, delayed
        X = Parallel(n_jobs=-1, batch_size=16)(delayed(_desc_one)((p, fn)) for p in df.path)
        np.savez_compressed(dst, X=np.stack(X), md5=df.md5.values, path=df.path.values)
        log("handcrafted:", np.stack(X).shape)
    import torch, timm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tags = {}
    for name, tname in CONFIG["BACKBONES"].items():
        dst = out("features", "frozen_%s.npz" % safe(name))
        if os.path.exists(dst): continue
        m = timm.create_model(tname, pretrained=CONFIG["PRETRAINED"], num_classes=0).to(dev).eval()
        tags[name] = _model_tag(m, tname)
        cfg = timm.data.resolve_data_config({}, model=m); tf = timm.data.create_transform(**cfg)
        feats = []
        with torch.no_grad():
            for k in range(0, len(units), 64):
                xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in units.path[k:k + 64]]).to(dev)
                with torch.autocast(dev, enabled=dev == "cuda"):
                    feats.append(m(xb).float().cpu().numpy())
        np.savez_compressed(dst, X=np.concatenate(feats), md5=units.md5.values)
        log("frozen", name, np.concatenate(feats).shape)
    if tags: json.dump(tags, open(out("features", "timm_tags.json"), "w"), indent=2)

def _model_tag(m, tname):
    pc = getattr(m, "pretrained_cfg", {}) or {}
    return dict(requested=tname, architecture=pc.get("architecture"), tag=pc.get("tag"),
                hf_hub_id=pc.get("hf_hub_id"), mean=pc.get("mean"), std=pc.get("std"), input_size=pc.get("input_size"))

def safe(s): return "".join(c if c.isalnum() else "_" for c in s)

def load_feats(name, md5s):
    z = np.load(out("features", name), allow_pickle=True)
    idx = {m: i for i, m in enumerate(z["md5"])}
    return z["X"][[idx[m] for m in md5s]]

# =============================================================================
# 3. LEAKAGE  (Table 5, reviewer 1 comment 5)
# =============================================================================
def heads():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    return {
        "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=CONFIG["LR_C"], max_iter=5000, class_weight="balanced")),
        "SVM": lambda: make_pipeline(StandardScaler(), SVC(C=CONFIG["SVM_C"], gamma=CONFIG["SVM_GAMMA"], class_weight="balanced")),
    }

def step_leakage():
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.metrics import f1_score
    df, units, folds = load_audit()
    z = np.load(out("features", "handcrafted_all.npz"), allow_pickle=True)
    Xall = z["X"]; d2 = df.collection.values == "D2"
    X, y, g = Xall[d2], df.type.values[d2], df.group.values[d2]
    u2 = units[units.collection == "D2"]
    Xu = load_feats("handcrafted_all.npz", u2.md5.values); yu = u2.type.values
    res = []
    for s in CONFIG["SEEDS"]:
        plans = {
            "Random image-level 5-fold, all images": (X, y, StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)),
            "Source-grouped 5-fold, all images": (X, y, StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g)),
            "Group-aware 5-fold, units": (Xu, yu, StratifiedKFold(5, shuffle=True, random_state=s).split(Xu, yu)),
        }
        for row, (XX, yy, it) in plans.items():
            for f, (tr, te) in enumerate(it):
                for hn, mk in heads().items():
                    m = mk().fit(XX[tr], yy[tr])
                    res.append(dict(row=row, head=hn, seed=s, fold=f, n_train=len(tr), n_test=len(te),
                                    macro_f1=f1_score(yy[te], m.predict(XX[te]), average="macro")))
            log("leakage", s, row)
    r = pd.DataFrame(res); r.to_csv(out("leakage", "leakage_runs.csv"), index=False)
    t = r.groupby(["row", "head"]).macro_f1.agg(["mean", "std", "count"]).reset_index()
    t.to_csv(out("leakage", "table5_rows.csv"), index=False); log("\n" + t.to_string())

# =============================================================================
# 4. PROBE  (reviewer 1 comment 6, reviewer 2 comment 3)
# =============================================================================
def step_probe():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    _, units, _ = load_audit()
    sets = {"Handcrafted": "handcrafted_all.npz"}
    for n in CONFIG["BACKBONES"]: sets["Frozen " + n] = "frozen_%s.npz" % safe(n)
    res = []
    for fs, fn in sets.items():
        for scope, U in [("all units", units), ("Basalt only", units[units.type == "Basalt"])]:
            X = load_feats(fn, U.md5.values); y = (U.collection.values == "D2").astype(int)
            for s in CONFIG["SEEDS"]:
                for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)):
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], y[tr])
                    res.append(dict(features=fs, scope=scope, seed=s, fold=f, n_test=len(te),
                                    auroc=roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
        log("probe", fs)
    r = pd.DataFrame(res); r.to_csv(out("probe", "probe_runs.csv"), index=False)
    t = r.groupby(["features", "scope"]).auroc.agg(["mean", "std"]).reset_index()
    t.to_csv(out("probe", "probe_table.csv"), index=False); log("\n" + t.to_string())

# =============================================================================
# 5. FINE-TUNING
# =============================================================================
def _torch_parts():
    import torch, timm
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T

    class DS(Dataset):
        def __init__(s, paths, ys, tf): s.p, s.y, s.tf = list(paths), list(ys), tf
        def __len__(s): return len(s.p)
        def __getitem__(s, i): return s.tf(Image.open(s.p[i]).convert("RGB")), s.y[i]

    class Net(nn.Module):
        def __init__(s, tname, K, fam):
            super().__init__()
            s.b = timm.create_model(tname, pretrained=CONFIG["PRETRAINED"], num_classes=0)
            d = getattr(s.b, 'head_hidden_size', None) or s.b.num_features; s.h = nn.Linear(d, K); s.fh = nn.Linear(d, 3) if fam else None
        def forward(s, x):
            z = s.b(x); return s.h(z), (s.fh(z) if s.fh is not None else None), z
    return torch, timm, nn, DS, DataLoader, T, Net

def train_eval(run_id, tname, train_df, val_df, eval_sets, classes, family_head=False, seed=0):
    """Train on train_df (columns path,type), return per-set dict of logits, features; also log curves."""
    dst = out("ft_runs", run_id + ".npz")
    if os.path.exists(dst):
        return dict(np.load(dst, allow_pickle=True))
    torch, timm, nn, DS, DataLoader, T, Net = _torch_parts()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K = len(classes); cidx = {c: i for i, c in enumerate(classes)}
    model = Net(tname, K, family_head).to(dev)
    pc = model.b.pretrained_cfg; mean, std = pc.get("mean", (0.485, 0.456, 0.406)), pc.get("std", (0.229, 0.224, 0.225))
    S = CONFIG["IMG_SIZE"]
    tf_tr = T.Compose([T.RandomResizedCrop(S, scale=CONFIG["CROP_SCALE"]), T.RandomHorizontalFlip(CONFIG["FLIP_P"]),
                       T.ToTensor(), T.Normalize(mean, std)])
    tf_ev = T.Compose([T.Resize(int(round(S / 0.875))), T.CenterCrop(S), T.ToTensor(), T.Normalize(mean, std)])
    ytr = np.array([cidx[t] for t in train_df.type])
    cnt = np.bincount(ytr, minlength=K).astype(np.float64)
    w = torch.tensor(len(ytr) / (K * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    fam_of = torch.tensor([FAMILIES.index(TAXONOMY[c][0]) for c in classes], device=dev)
    dl = DataLoader(DS(train_df.path, ytr, tf_tr), batch_size=CONFIG["BATCH"], shuffle=True,
                    num_workers=CONFIG["NUM_WORKERS"], drop_last=len(ytr) > CONFIG["BATCH"], pin_memory=dev == "cuda")
    heads_p = list(model.h.parameters()) + (list(model.fh.parameters()) if model.fh is not None else [])
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": CONFIG["LR_BACKBONE"]},
                             {"params": heads_p, "lr": CONFIG["LR_HEAD"]}], weight_decay=CONFIG["WD"])
    E = CONFIG["EPOCHS"]; steps = E * len(dl); warm = CONFIG["WARMUP_EPOCHS"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / max(1, warm) if k < warm
                                              else 0.5 * (1 + math.cos(math.pi * (k - warm) / max(1, steps - warm))))
    use_amp = dev == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce = nn.CrossEntropyLoss(weight=w); ce_f = nn.CrossEntropyLoss()
    yva = np.array([cidx[t] for t in val_df.type]) if len(val_df) else None
    curve = []
    def run_eval(paths):
        model.eval(); L, Z = [], []
        dle = DataLoader(DS(paths, [0] * len(paths), tf_ev), batch_size=128, num_workers=CONFIG["NUM_WORKERS"])
        with torch.no_grad():
            for xb, _ in dle:
                with torch.autocast(dev, enabled=use_amp):
                    lo, _, z = model(xb.to(dev))
                L.append(lo.float().cpu().numpy()); Z.append(z.float().cpu().numpy())
        return (np.concatenate(L), np.concatenate(Z)) if L else (np.zeros((0, K)), np.zeros((0, 1)))
    for ep in range(E):
        model.train(); tl, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            with torch.autocast(dev, enabled=use_amp):
                lo, fl, _ = model(xb); loss = ce(lo, yb)
                if fl is not None: loss = loss + CONFIG["FAMILY_LAMBDA"] * ce_f(fl, fam_of[yb])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nb += 1
        row = dict(epoch=ep + 1, train_loss=tl / max(nb, 1))
        if yva is not None and len(yva):
            lv, _ = run_eval(val_df.path)
            import torch.nn.functional as F
            row["val_loss"] = float(F.cross_entropy(torch.tensor(lv), torch.tensor(yva)).item())
            from sklearn.metrics import f1_score
            row["val_macro_f1"] = f1_score(yva, lv.argmax(1), average="macro")
        curve.append(row)
    pd.DataFrame(curve).to_csv(out("ft_curves", run_id + ".csv"), index=False)
    res = {"classes": np.array(classes)}
    for nm, d in list(eval_sets.items()) + [("train", train_df), ("val", val_df)]:
        L, Z = run_eval(d.path); res[nm + "_logits"] = L; res[nm + "_feat"] = Z.astype(np.float16)
        res[nm + "_md5"] = d.md5.values; res[nm + "_type"] = d.type.values
    res["tag"] = json.dumps(_model_tag(model.b, tname))
    np.savez_compressed(dst, **res)
    log("done", run_id, curve[-1])
    del model; torch.cuda.empty_cache() if dev == "cuda" else None
    return dict(np.load(dst, allow_pickle=True))

def split_val(tr, seed):
    from sklearn.model_selection import train_test_split
    if CONFIG["VAL_FRACTION"] <= 0: return tr, tr.iloc[:0]
    a, b = train_test_split(tr, test_size=CONFIG["VAL_FRACTION"], random_state=seed, stratify=tr.type)
    return a, b

def fold_units(units, folds, seed, fold):
    f = folds[folds.seed == seed][["md5", "fold"]]
    u = units.merge(f, on="md5")
    return u[u.fold != fold], u[u.fold == fold]

def _dbg(df, n=6):
    return df.groupby("type", group_keys=False).head(n) if CONFIG["FAST_DEBUG"] else df

def all_backbones():
    b = dict(CONFIG["BACKBONES"])
    if CONFIG["COMPARATOR"]: b.update(CONFIG["COMPARATOR"])
    return b

def step_finetune(which="all"):
    _, units, folds = load_audit()
    BB = all_backbones()
    # (a) five-fold closed-set P1 for leading models + comparator
    if which in ("all", "ft5"):
        models = list(CONFIG["FT5_MODELS"]) + ([(k, False) for k in CONFIG["COMPARATOR"]] if CONFIG["COMPARATOR"] else [])
        s = CONFIG["FT5_SEED"]
        for name, fam in models:
            for f in range(CONFIG["NFOLD"]):
                tr, te = fold_units(units, folds, s, f); tr, va = split_val(tr, s)
                rid = "P1_%s%s_s%d_f%d" % (safe(name), "_fam" if fam else "", s, f)
                train_eval(rid, BB[name], _dbg(tr), _dbg(va), {"test": _dbg(te)}, TYPES, fam, s)
    # (b) cross-collection O1/O2 (fold 0, three seeds), all seven scores, with validation split
    if which in ("all", "osr"):
        for name in CONFIG["OSR_BACKBONES"]:
            for s in CONFIG["SEEDS"]:
                for proto, C, O in [("O1", "D1", "D2"), ("O2", "D2", "D1")]:
                    uc = units[units.collection == C]
                    tr, te = fold_units(uc, folds, s, 0); tr, va = split_val(tr, s)
                    other = units[units.collection == O]
                    ev = {"test": _dbg(te), "unknown": _dbg(other[other.type != "Basalt"]), "otherbasalt": _dbg(other[other.type == "Basalt"])}
                    cls = sorted(uc.type.unique(), key=TYPES.index)
                    train_eval("%s_%s_s%d" % (proto, safe(name), s), BB[name], _dbg(tr), _dbg(va), ev, cls, False, s)
    # (c) within-collection hold-out P3b (fold 0), also used as the validation proxy for score selection
    if which in ("all", "p3b"):
        for name in CONFIG["OSR_BACKBONES"]:
            for s in CONFIG["P3B_SEEDS"]:
                for C, splits in CONFIG["P3B_SPLITS"].items():
                    uc = units[units.collection == C]
                    for k, held in enumerate(splits):
                        tr_all, te_all = fold_units(uc, folds, s, 0)
                        known_tr = tr_all[~tr_all.type.isin(held)]; tr, va = split_val(known_tr, s)
                        ev = {"test": _dbg(te_all[~te_all.type.isin(held)]), "unknown": _dbg(te_all[te_all.type.isin(held)]),
                              "valunknown": _dbg(tr_all[tr_all.type.isin(held)])}
                        cls = sorted(known_tr.type.unique(), key=TYPES.index)
                        train_eval("P3b_%s_%s_k%d_s%d" % (C, safe(name), k, s), BB[name], _dbg(tr), _dbg(va), ev, cls, False, s)

# =============================================================================
# 6. ANALYSIS
# =============================================================================
SCORES = ["MSP", "MaxLogit", "Energy", "Mahalanobis", "kNN-1", "kNN-5", "OC-SVM"]

def osr_scores(R, sets):
    """Return dict set -> dict score -> array (higher = more likely known)."""
    from scipy.special import logsumexp, softmax
    from sklearn.covariance import LedoitWolf
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    Ztr = R["train_feat"].astype(np.float32); ytr = R["train_type"]; cls = list(R["classes"])
    mu = np.stack([Ztr[ytr == c].mean(0) for c in cls])
    lw = LedoitWolf().fit(Ztr - mu[[cls.index(c) for c in ytr]]); P = lw.precision_
    nrm = lambda Z: Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    Ntr = nrm(Ztr)
    sc = StandardScaler().fit(Ztr); oc = OneClassSVM(nu=0.1, gamma="scale").fit(sc.transform(Ztr))
    out_ = {}
    for s in sets:
        L, Z = R[s + "_logits"], R[s + "_feat"].astype(np.float32)
        if len(L) == 0: continue
        d = {"MSP": softmax(L, 1).max(1), "MaxLogit": L.max(1), "Energy": logsumexp(L, 1)}
        dm = np.stack([np.einsum("ij,jk,ik->i", Z - m, P, Z - m) for m in mu], 1); d["Mahalanobis"] = -dm.min(1)
        sim = nrm(Z) @ Ntr.T; dist = np.sqrt(np.maximum(2 - 2 * sim, 0)); dist.sort(1)
        d["kNN-1"] = -dist[:, 0]; d["kNN-5"] = -dist[:, min(4, dist.shape[1] - 1)]
        d["OC-SVM"] = oc.decision_function(sc.transform(Z))
        out_[s] = d
    return out_

def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])

def fpr_at(known, unknown, thr): return float((unknown >= thr).mean())
def t95(known): return np.quantile(known, 0.05)

def oscr(known_s, known_correct, unk_s):
    thr = np.unique(np.r_[known_s, unk_s])[::-1]
    ccr = [((known_s >= t) & known_correct).mean() for t in thr]; fpr = [(unk_s >= t).mean() for t in thr]
    return float(getattr(np, "trapezoid", getattr(np, "trapz", None))(np.r_[0, ccr, ccr[-1] if ccr else 0], np.r_[0, fpr, 1]))

def closed_metrics(y, p):
    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
    fam = lambda a: np.array([TAXONOMY[t][0] for t in a])
    err = y != p
    d = np.array([tax_distance(a, b) for a, b in zip(y[err], p[err])])
    return dict(accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"),
                balanced_acc=balanced_accuracy_score(y, p), family_acc=(fam(y) == fam(p)).mean(),
                cross_family_share=(d == 3).mean() if len(d) else np.nan, severity=d.mean() if len(d) else np.nan)

def step_analyse():
    from scipy.stats import wilcoxon, spearmanr
    rng = np.random.default_rng(0)
    # ---------- five-fold P1
    rows, preds = [], []
    for f in sorted(glob.glob(out("ft_runs", "P1_*.npz"))):
        R = dict(np.load(f, allow_pickle=True)); rid = os.path.basename(f)[:-4]
        cls = np.array(R["classes"]); y = R["test_type"]; p = cls[R["test_logits"].argmax(1)]
        model = rid[3:rid.rindex("_s")]; fold = int(rid.split("_f")[-1])
        rows.append(dict(model=model, fold=fold, **closed_metrics(y, p)))
        preds.append(pd.DataFrame(dict(model=model, fold=fold, md5=R["test_md5"], true=y, pred=p)))
        cv = pd.read_csv(out("ft_curves", rid + ".csv"))
        rows[-1].update(final_val_loss=cv.val_loss.iloc[-1] if "val_loss" in cv else np.nan,
                        min_val_loss=cv.val_loss.min() if "val_loss" in cv else np.nan,
                        val_loss_change_last3=(cv.val_loss.iloc[-1] - cv.val_loss.iloc[-3]) if "val_loss" in cv and len(cv) >= 3 else np.nan)
    if rows:
        r = pd.DataFrame(rows); r.to_csv(out("results", "p1_5fold_runs.csv"), index=False)
        t = r.groupby("model").agg(["mean", "std"]); t.to_csv(out("results", "p1_5fold_table.csv"))
        pr = pd.concat(preds); pr.to_csv(out("results", "p1_5fold_predictions.csv"), index=False)
        # cluster bootstrap: resample units (one unit = one leakage group) within each test fold
        bo = []
        for m, g in pr.groupby("model"):
            fs = [x for _, x in g.groupby("fold")]; vals = []
            for _ in range(CONFIG["N_BOOT"]):
                from sklearn.metrics import f1_score
                v = []
                for x in fs:
                    ix = rng.integers(0, len(x), len(x)); v.append(f1_score(x.true.values[ix], x.pred.values[ix], average="macro"))
                vals.append(np.mean(v))
            bo.append(dict(model=m, metric="macro_f1", point=r[r.model == m].macro_f1.mean(),
                           ci_low=np.percentile(vals, 2.5), ci_high=np.percentile(vals, 97.5)))
        pd.DataFrame(bo).to_csv(out("results", "bootstrap_p1.csv"), index=False)
        log("P1 5-fold\n" + r.groupby("model")[["macro_f1", "family_acc", "cross_family_share"]].agg(["mean", "std"]).to_string())
    # ---------- P3b (fine-tuned) + validation AUROC of each score with pseudo-unknowns
    p3 = []
    for f in sorted(glob.glob(out("ft_runs", "P3b_*.npz"))):
        R = dict(np.load(f, allow_pickle=True)); rid = os.path.basename(f)[:-4]
        _, C, *mid, k, s = rid.split("_"); name = "_".join(mid); k = int(k[1:]); s = int(s[1:])
        S = osr_scores(R, ["test", "unknown", "val", "valunknown"])
        held = CONFIG["P3B_SPLITS"][C][k]; known = list(R["classes"])
        dmin = min(tax_distance(h, q) for h in held for q in known)
        for sc in SCORES:
            p3.append(dict(collection=C, backbone=name, split=k, seed=s, held_out="|".join(held), min_tax_distance=dmin,
                           score=sc, test_auroc=auroc(S["test"][sc], S["unknown"][sc]),
                           test_fpr95=fpr_at(None, S["unknown"][sc], t95(S["test"][sc])),
                           val_auroc=auroc(S["val"][sc], S["valunknown"][sc]) if "valunknown" in S and "val" in S else np.nan))
    if p3:
        p3 = pd.DataFrame(p3); p3.to_csv(out("results", "p3b_ft_all.csv"), index=False)
        best = p3.groupby(["collection", "backbone", "score"])[["test_auroc", "test_fpr95"]].mean().reset_index()
        best = best.loc[best.groupby(["collection", "backbone"]).test_auroc.idxmax()]
        best.to_csv(out("results", "p3b_ft_best_posthoc.csv"), index=False)
        p3[p3.score == "MSP"].groupby(["collection", "split", "held_out", "min_tax_distance"]).test_auroc.mean().reset_index() \
            .to_csv(out("results", "p3b_distance_control_MSP.csv"), index=False)
        p3.groupby(["collection", "split", "held_out", "min_tax_distance", "backbone"]).test_auroc.max().reset_index() \
            .to_csv(out("results", "p3b_distance_control_best.csv"), index=False)
        # validation-based score choice per (collection, backbone): highest mean validation AUROC
        sel = p3.groupby(["collection", "backbone", "score"]).val_auroc.mean().reset_index()
        sel = sel.loc[sel.groupby(["collection", "backbone"]).val_auroc.idxmax()].rename(columns={"score": "selected_score"})
        sel.to_csv(out("results", "score_selection_validation.csv"), index=False)
    else:
        sel = None
    # ---------- O1/O2: all scores on untouched test units, thresholds from validation known units
    oo, pair = [], []
    for f in sorted(glob.glob(out("ft_runs", "O[12]_*.npz"))):
        R = dict(np.load(f, allow_pickle=True)); rid = os.path.basename(f)[:-4]
        proto = rid[:2]; name = rid[3:rid.rindex("_s")]; s = int(rid.split("_s")[-1])
        S = osr_scores(R, ["test", "unknown", "otherbasalt", "val"])
        cls = np.array(R["classes"]); correct = cls[R["test_logits"].argmax(1)] == R["test_type"]
        for sc in SCORES:
            k_, u_, b_ = S["test"][sc], S["unknown"][sc], S.get("otherbasalt", {}).get(sc, np.array([]))
            thr_val = t95(S["val"][sc]) if "val" in S else np.nan
            oo.append(dict(protocol=proto, backbone=name, seed=s, score=sc, auroc=auroc(k_, u_),
                           fpr95_test_threshold=fpr_at(None, u_, t95(k_)),
                           fpr_at_val_threshold=fpr_at(None, u_, thr_val), tpr_known_at_val_threshold=float((k_ >= thr_val).mean()),
                           oscr=oscr(k_, correct, u_), B_U=auroc(b_, u_), K_B=auroc(k_, b_)))
    if oo:
        oo = pd.DataFrame(oo); oo.to_csv(out("results", "osr_ft_all_scores.csv"), index=False)
        mean = oo.groupby(["protocol", "backbone", "score"]).mean(numeric_only=True).drop(columns="seed").reset_index()
        mean.to_csv(out("results", "osr_ft_mean.csv"), index=False)
        ph = mean.loc[mean.groupby(["protocol", "backbone"]).auroc.idxmax()][["protocol", "backbone", "score", "auroc"]] \
            .rename(columns={"score": "posthoc_best", "auroc": "posthoc_auroc"})
        if sel is not None:
            m2 = sel.assign(protocol=sel.collection.map({"D1": "O1", "D2": "O2"}))
            v = m2.merge(mean, left_on=["protocol", "backbone", "selected_score"], right_on=["protocol", "backbone", "score"])
            v = v.merge(ph, on=["protocol", "backbone"]); v["matches_posthoc"] = v.selected_score == v.posthoc_best
            v.to_csv(out("results", "osr_validation_selected.csv"), index=False)
            log("validation-selected\n" + v[["protocol", "backbone", "selected_score", "auroc", "fpr_at_val_threshold", "posthoc_best", "posthoc_auroc"]].to_string())
        # Wilcoxon vs MSP over paired runs (backbone x seed), Holm within protocol
        hl = []
        for proto, g in oo.groupby("protocol"):
            piv = g.pivot_table(index=["backbone", "seed"], columns="score", values="auroc")
            tmp = []
            for sc in SCORES:
                if sc == "MSP" or sc not in piv: continue
                dif = piv[sc] - piv["MSP"]
                p = wilcoxon(dif).pvalue if (dif != 0).any() else 1.0
                tmp.append(dict(protocol=proto, score=sc, mean_diff_vs_MSP=dif.mean(), n_pairs=len(dif), p_raw=p))
            tmp = sorted(tmp, key=lambda d: d["p_raw"]); m = len(tmp); run = 0
            for i, d in enumerate(tmp):
                run = max(run, min(1.0, (m - i) * d["p_raw"])); d["p_holm"] = run
            hl += tmp
        pd.DataFrame(hl).to_csv(out("results", "holm_scores_vs_msp.csv"), index=False)
        # bootstrap CI for AUROC of each backbone's validation-selected (else MSP) score: resample known and unknown units, and seeds
        bo = []
        for (proto, name), g in oo.groupby(["protocol", "backbone"]):
            sc = "MSP"
            if sel is not None:
                q = sel[(sel.collection == {"O1": "D1", "O2": "D2"}[proto]) & (sel.backbone == name)]
                if len(q): sc = q.selected_score.iloc[0]
            runs = []
            for f in sorted(glob.glob(out("ft_runs", "%s_%s_s*.npz" % (proto, name)))):
                S = osr_scores(dict(np.load(f, allow_pickle=True)), ["test", "unknown"]); runs.append((S["test"][sc], S["unknown"][sc]))
            vals = []
            for _ in range(min(CONFIG["N_BOOT"], 1000)):
                k_, u_ = runs[rng.integers(0, len(runs))]
                vals.append(auroc(k_[rng.integers(0, len(k_), len(k_))], u_[rng.integers(0, len(u_), len(u_))]))
            bo.append(dict(protocol=proto, backbone=name, score=sc, point=g[g.score == sc].auroc.mean(),
                           ci_low=np.percentile(vals, 2.5), ci_high=np.percentile(vals, 97.5)))
        pd.DataFrame(bo).to_csv(out("results", "bootstrap_osr.csv"), index=False)
    # model identifiers for Table 3
    tags = {}
    for f in glob.glob(out("ft_runs", "*.npz")):
        R = np.load(f, allow_pickle=True)
        if "tag" in R: t = json.loads(str(R["tag"])); tags[t["requested"]] = t
    json.dump(tags, open(out("results", "timm_tags_finetuned.json"), "w"), indent=2)
    json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()}, open(out("results", "config_used.json"), "w"), indent=2, default=str)
    import importlib
    vers = {}
    for m in ["torch", "torchvision", "timm", "sklearn", "skimage", "lightgbm", "numpy", "scipy", "pandas", "PIL"]:
        try: vers[m] = importlib.import_module(m).__version__
        except Exception: pass
    vers["python"] = sys.version.split()[0]
    json.dump(vers, open(out("results", "software_versions.json"), "w"), indent=2)
    log("analysis written to", out("results", ""))

def step_dhash_eval(labeled_csv):
    d = pd.read_csv(labeled_csv); d = d[d.same_photo.astype(str).isin(["0", "1"])]; d["same_photo"] = d.same_photo.astype(int)
    T = CONFIG["DHASH_BITS_THRESHOLD"]; linked = d.hamming <= T
    r = dict(labelled_pairs=len(d), threshold=T,
             precision_linked_pairs_same=float(d[linked].same_photo.mean()) if linked.any() else np.nan,
             recall_in_sample=float(linked[d.same_photo == 1].mean()) if (d.same_photo == 1).any() else np.nan,
             same_photo_rate_by_bin=d.assign(b=pd.cut(d.hamming, [-1, 5, 10, 15, 20, 25, 30, 40])).groupby("b", observed=True).same_photo.mean().to_dict())
    r["same_photo_rate_by_bin"] = {str(k): v for k, v in r["same_photo_rate_by_bin"].items()}
    json.dump(r, open(out("audit", "dhash_validation.json"), "w"), indent=2); log(r)

# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["audit", "features", "leakage", "probe", "finetune", "analyse", "dhash_eval", "all"])
    ap.add_argument("--which", default="all", help="finetune subset: all | ft5 | osr | p3b")
    ap.add_argument("--labeled", default=None)
    ap.add_argument("--config", default=None, help="JSON file overriding CONFIG")
    a = ap.parse_args()
    if a.config: CONFIG.update(json.load(open(a.config)))
    if a.step in ("audit", "all"): step_audit()
    if a.step in ("features", "all"): step_features()
    if a.step in ("leakage", "all"): step_leakage()
    if a.step in ("probe", "all"): step_probe()
    if a.step in ("finetune", "all"): step_finetune(a.which)
    if a.step in ("analyse", "all"): step_analyse()
    if a.step == "dhash_eval": step_dhash_eval(a.labeled)
