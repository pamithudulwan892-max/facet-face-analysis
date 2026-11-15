"""Train the face score model on SCUT-FBP5500 features and export it for the site.

Steps
1. Load the features the browser extracted (landmarks, skin stats, embeddings).
2. Check the left/right landmark pairs against the mean face shape.
3. Build geometry features with the same JavaScript the site uses (Node).
4. Compare a few models on the official 60/40 split and with 5-fold CV.
5. Refit the chosen model on all data and write site/models/model.json.

Run from the project root:  python tools/train.py
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "SCUT-FBP5500_v2")
OUT = os.path.join(ROOT, "site", "models", "model.json")
REPORT = os.path.join(ROOT, "data", "training_report.json")
sys.path.insert(0, ROOT)

PIXEL_MODEL = ["skin_texture", "skin_tone_var", "skin_redness", "redness_var", "under_eye_dark"]
PIXEL_ALL = PIXEL_MODEL + ["brightness", "contrast", "sharpness", "light_balance"]
BALANCE = ["face_ratio", "jaw_ratio", "eye_width", "eye_spacing", "nose_width", "nose_length",
           "mouth_width", "lip_fullness", "thirds_ratio", "chin_length", "brow_height"]
MIDLINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 164, 0, 17, 18, 200, 199, 175, 152]


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def load_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "landmarks", "batch_*.json"))):
        with open(f) as fh:
            rows.extend(json.load(fh))
    return rows


def normalize(lm):
    P = np.asarray(lm)[:, :2]
    r = (P[33] + P[133]) / 2
    l = (P[362] + P[263]) / 2
    ang = np.arctan2(l[1] - r[1], l[0] - r[0])
    c, s = np.cos(-ang), np.sin(-ang)
    d = P - (r + l) / 2
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1) / np.linalg.norm(l - r)


def check_pairs(rows):
    """Keep only pairs that mirror each other well on the average face."""
    mean = np.mean([normalize(r["lm"]) for r in rows], axis=0)
    x0 = mean[MIDLINE, 0].mean()
    from_js = subprocess.run(
        ["node", "-e", "import('./site/js/features.js').then(m=>console.log(JSON.stringify(m.PAIRS)))"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    pairs = json.loads(from_js)
    good, bad = [], []
    for a, b in pairs:
        e = np.hypot(mean[a, 0] - (2 * x0 - mean[b, 0]), mean[a, 1] - mean[b, 1])
        (good if e < 0.04 else bad).append(([a, b], round(float(e), 4)))
    print(f"pairs: {len(good)} good, {len(bad)} rejected {bad}")
    return [p for p, _ in good], mean


def main():
    rows = load_rows()
    labels = pd.read_csv(os.path.join(RAW, "train_test_files", "All_labels.txt"),
                         sep=r"\s+", header=None, names=["file", "rating"])
    print(f"{len(rows)} faces extracted, {len(labels)} labels")

    pairs, mean_shape = check_pairs(rows)
    pairs_file = os.path.join(ROOT, "data", "pairs.json")
    with open(pairs_file, "w") as fh:
        json.dump(pairs, fh)
    subprocess.run(["node", "tools/build_geometry.mjs", pairs_file], cwd=ROOT, check=True)

    geo = pd.read_csv(os.path.join(ROOT, "data", "geometry.csv"))
    shape = pd.read_csv(os.path.join(ROOT, "data", "shape.csv"), header=None).set_index(0)
    base = pd.DataFrame({"file": [r["file"] for r in rows]})
    px = pd.DataFrame([r["px"] for r in rows])
    pose = pd.DataFrame([r["pose"] for r in rows])
    emb = np.array([r["emb"] for r in rows], dtype=np.float32)

    df = pd.concat([base, px, pose], axis=1).merge(geo, on="file").merge(labels, on="file")
    order = df["file"].map({f: i for i, f in enumerate(base["file"])}).to_numpy()
    emb = emb[order]
    S = shape.loc[df["file"]].to_numpy(dtype=np.float32)
    geo_names = [c for c in geo.columns if c != "file"]
    y = df["rating"].to_numpy()
    print("pose on dataset: yaw mean %.1f sd %.1f, noseRatio mean %.3f sd %.3f" % (
        df.yaw.mean(), df.yaw.std(), df.noseRatio.mean(), df.noseRatio.std()))

    blocks = {
        "embedding": emb,
        "shape": S,
        "geometry": df[geo_names].to_numpy(),
        "skin": df[PIXEL_MODEL].to_numpy(),
    }

    split_dir = os.path.join(RAW, "train_test_files", "split_of_60%training and 40%testing")
    tr_files = set(pd.read_csv(os.path.join(split_dir, "train.txt"), sep=r"\s+", header=None)[0])
    is_tr = df["file"].isin(tr_files).to_numpy()

    def build(names):
        return np.hstack([blocks[n] for n in names])

    def evaluate(names, alpha):
        X = build(names)
        sc = StandardScaler().fit(X[is_tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[is_tr]), y[is_tr])
        p = m.predict(sc.transform(X[~is_tr]))
        return p

    results = {}
    candidates = [
        ["geometry"], ["skin"], ["geometry", "skin"], ["shape"], ["shape", "geometry", "skin"],
        ["embedding"], ["embedding", "shape", "geometry", "skin"],
    ]
    best = None
    for names in candidates:
        n_feat = build(names).shape[1]
        for alpha in [1, 10, 30, 100, 300, 1000, 3000, 10000]:
            if n_feat < 50 and alpha > 100:
                continue
            # choose alpha with 5-fold CV inside the training part only
            X = build(names)[is_tr]
            yt = y[is_tr]
            scores = []
            for a, b in KFold(5, shuffle=True, random_state=0).split(X):
                sc = StandardScaler().fit(X[a])
                m = Ridge(alpha=alpha).fit(sc.transform(X[a]), yt[a])
                scores.append(pcc(m.predict(sc.transform(X[b])), yt[b]))
            cv = float(np.mean(scores))
            key = "+".join(names)
            if key not in results or cv > results[key]["cv_pcc"]:
                results[key] = {"alpha": alpha, "cv_pcc": round(cv, 4), "n_features": n_feat}
        r = results["+".join(names)]
        p = evaluate(names, r["alpha"])
        r.update(test_pcc=round(pcc(p, y[~is_tr]), 4),
                 test_mae=round(float(mean_absolute_error(y[~is_tr], p)), 4),
                 test_rmse=round(float(np.sqrt(mean_squared_error(y[~is_tr], p))), 4))
        print("+".join(names).ljust(40), r)
        if best is None or r["cv_pcc"] > results[best]["cv_pcc"]:
            best = "+".join(names)

    names = best.split("+")
    alpha = results[best]["alpha"]
    print("chosen:", best, results[best])

    # Per-group accuracy on the held-out 40% (for the bias section of the site).
    p_test = evaluate(names, alpha)
    groups = {}
    test_files = df["file"][~is_tr].str[:2].to_numpy()
    names_map = {"AF": "Asian female", "AM": "Asian male", "CF": "Caucasian female", "CM": "Caucasian male"}
    for g, label in names_map.items():
        m = test_files == g
        if m.sum() < 2:
            continue
        groups[label] = {"n": int(m.sum()), "pcc": round(pcc(p_test[m], y[~is_tr][m]), 3),
                         "mae": round(float(mean_absolute_error(y[~is_tr][m], p_test[m])), 3)}
    print(groups)

    # Out-of-fold predictions on all faces give a fair distribution for percentiles.
    X = build(names)
    oof = np.zeros(len(y))
    for a, b in KFold(5, shuffle=True, random_state=1).split(X):
        sc = StandardScaler().fit(X[a])
        oof[b] = Ridge(alpha=alpha).fit(sc.transform(X[a]), y[a]).predict(sc.transform(X[b]))
    print("5-fold out-of-fold PCC on all data: %.4f, MAE %.4f" % (pcc(oof, y), mean_absolute_error(y, oof)))

    # Final model on all 5500 faces, folded into plain weights + bias.
    sc = StandardScaler().fit(X)
    final = Ridge(alpha=alpha).fit(sc.transform(X), y)
    w = final.coef_ / sc.scale_
    b = final.intercept_ - float(np.sum(final.coef_ * sc.mean_ / sc.scale_))

    q = np.linspace(0, 1, 101)
    feat_all = df[geo_names + PIXEL_ALL]
    bal_mean = df[BALANCE].mean()
    bal_std = df[BALANCE].std()
    bal_dist = np.sqrt((((df[BALANCE] - bal_mean) / bal_std) ** 2).mean(axis=1))

    model = {
        "version": 1,
        "inputs": names,
        "sizes": {n: int(blocks[n].shape[1]) for n in names},
        "geometry_names": geo_names,
        "skin_names": PIXEL_MODEL,
        "weights": [round(float(v), 7) for v in w],
        "bias": round(b, 6),
        "pairs": pairs,
        "rating_range": [1, 5],
        "rating_quantiles": [round(float(v), 4) for v in np.quantile(oof, q)],
        "label_quantiles": [round(float(v), 4) for v in np.quantile(y, q)],
        "feature_quantiles": {c: [round(float(v), 5) for v in np.quantile(feat_all[c], q)] for c in feat_all.columns},
        "balance": {"names": BALANCE, "mean": bal_mean.round(5).tolist(), "std": bal_std.round(5).tolist(),
                    "quantiles": [round(float(v), 4) for v in np.quantile(bal_dist, q)]},
        "pose_reference": {
            "noseRatio": round(float(df.noseRatio.mean()), 4),
            "noseRatio_p90": round(float(np.quantile(np.abs(df.noseRatio - df.noseRatio.mean()), 0.9)), 4),
            "yaw_p50": round(float(np.quantile(np.abs(df.yaw), 0.5)), 2),
            "yaw_p90": round(float(np.quantile(np.abs(df.yaw), 0.9)), 2),
        },
        "metrics": {
            "chosen": best,
            "alpha": alpha,
            "test_split": "official 60% train / 40% test",
            "test_pcc": results[best]["test_pcc"],
            "test_mae": results[best]["test_mae"],
            "test_rmse": results[best]["test_rmse"],
            "cv5_pcc_all": round(pcc(oof, y), 4),
            "cv5_mae_all": round(float(mean_absolute_error(y, oof)), 4),
            "n_faces": int(len(y)),
            "groups": groups,
            "baselines": results,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(model, fh, separators=(",", ":"))
    with open(REPORT, "w") as fh:
        json.dump(model["metrics"], fh, indent=2)
    print(f"saved {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
