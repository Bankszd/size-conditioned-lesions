"""Task Y, step 1: paired size gap S for existing AP arms, one shared resample set.

Reads the Task C per-image JSONL (all arms for an image live in one row, so a
single image resample applies to every arm simultaneously). Builds the COCO
evalImgs cache once per (version, arm), then re-aggregates AP / APs / APl for
each of N_BOOT whole-image percentile resamples.
"""

# --- release path handling -------------------------------------------------
# The frozen scripts were written inside the authors' working directory.  For
# the release every absolute location is resolved from the environment; the
# defaults reproduce the layout described in docs/REPRODUCE.md.
import os as _os
DATA_ROOT = _os.environ.get("DATA_ROOT", "data")
OUT_ROOT = _os.environ.get("OUT_ROOT", "out")
CKPT_ROOT = _os.environ.get("CKPT_ROOT", "weights")
WORK_ROOT = _os.environ.get("WORK_ROOT", "work")
REPO_ROOT = _os.environ.get("REPO_ROOT", ".")


def _rel(root, tail):
    return _os.path.join(root, tail) if tail else root

# --- end release path handling ---------------------------------------------
import argparse, gzip, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANN = "_rel(DATA_ROOT, "")/coco/annotations/instances_val2017.json"
HOLDOUT_SHA = "dab7feec54b51a9f6bad9c1dc25c48f093be83d7b5f7cd15f44adf62fbd036ac"
BOOT_SEED = 20271004

def ap_from_matches(evaluator, image_draw, area_index=0, max_det=100):
    params = evaluator._paramsEval
    image_count = len(params.imgIds)
    area_count = len(params.areaRng)
    iou_count = len(params.iouThrs)
    recall_count = len(params.recThrs)
    category_values = []
    for category_index in range(len(params.catIds)):
        offset = (category_index * area_count + area_index) * image_count
        image_records = evaluator.evalImgs[offset:offset + image_count]
        records = [image_records[i] for i in image_draw]
        records = [r for r in records if r is not None]
        if not records:
            continue
        gt_ignore = np.concatenate([r["gtIgnore"] for r in records])
        positive_gt = np.count_nonzero(gt_ignore == 0)
        if positive_gt == 0:
            continue
        scores = np.concatenate([r["dtScores"][:max_det] for r in records])
        sorting = np.argsort(-scores, kind="mergesort")
        matches = np.concatenate([r["dtMatches"][:, :max_det] for r in records], axis=1)[:, sorting]
        ignored = np.concatenate([r["dtIgnore"][:, :max_det] for r in records], axis=1)[:, sorting]
        tp = np.logical_and(matches, np.logical_not(ignored))
        fp = np.logical_and(np.logical_not(matches), np.logical_not(ignored))
        tcs = np.cumsum(tp, axis=1).astype(float)
        fcs = np.cumsum(fp, axis=1).astype(float)
        precision = np.zeros((iou_count, recall_count), dtype=float)
        for i, (trow, frow) in enumerate(zip(tcs, fcs)):
            recall = trow / positive_gt
            values = trow / (trow + frow + np.spacing(1))
            if len(values):
                values = np.maximum.accumulate(values[::-1])[::-1]
                pos = np.searchsorted(recall, params.recThrs, side="left")
                valid = pos < len(values)
                precision[i, valid] = values[pos[valid]]
        category_values.append(precision.mean())
    return float(np.mean(category_values))

EV = {}
DRAWS = None
KEYS = None

def _init(ev, draws, keys):
    global EV, DRAWS, KEYS
    EV, DRAWS, KEYS = ev, draws, keys

def _work(idx):
    out = {}
    draw = DRAWS[idx]
    for key in KEYS:
        ev = EV[key]
        out[key] = (0.0, ap_from_matches(ev, draw, 1), ap_from_matches(ev, draw, 3))
    return idx, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--arms", default="clean,in4,in_m0.02")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out-dir", default="_rel(OUT_ROOT, "")")
    ap.add_argument("--draws", default="_rel(OUT_ROOT, "")/task_y_draws_2000.npy")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    versions = ("layer2", "all_layers")
    t0 = time.time()

    coco = COCO(ANN)
    ids = sorted(int(x) for x in np.random.Generator(np.random.PCG64(20270925))
                 .permutation(sorted(coco.getImgIds()))[1000:5000])
    import hashlib
    assert hashlib.sha256(",".join(str(x) for x in ids).encode()).hexdigest() == HOLDOUT_SHA
    print("HOLDOUT_OK", len(ids), flush=True)

    dets = {}
    with gzip.open(args.src, "rt") as f:
        for line in f:
            row = json.loads(line)
            for version, dd in row["arms"].items():
                for arm, d in dd.items():
                    if arm in arms:
                        dets.setdefault((version, arm), []).extend(d)
    print("LOADED_DETS", json.dumps({str(k): len(v) for k, v in dets.items()}), "%.0fs" % (time.time() - t0), flush=True)

    keys = []
    for version in versions:
        for arm in arms:
            if (version, arm) not in dets:
                print("MISSING_ARM", version, arm, flush=True)
                continue
            dt = coco.loadRes(dets[(version, arm)])
            e = COCOeval(coco, dt, "bbox")
            e.params.imgIds = ids
            e.evaluate(); e.accumulate()
            EV[(version, arm)] = e
            keys.append((version, arm))
            print("EVALUATED", version, arm, "%.1fs" % (time.time() - t0), flush=True)
            del dt

    if os.path.exists(args.draws):
        draws = np.load(args.draws)[:args.n_boot]
        print("DRAWS_REUSED", draws.shape, flush=True)
    else:
        draws = np.random.Generator(np.random.PCG64(BOOT_SEED)).integers(
            0, len(ids), size=(args.n_boot, len(ids)), dtype=np.int32)
        np.save(args.draws, draws)
        print("DRAWS_CREATED", draws.shape, flush=True)

    identity = np.arange(len(ids))
    obs = {}
    for key in keys:
        obs[key] = (ap_from_matches(EV[key], identity, 0),
                    ap_from_matches(EV[key], identity, 1),
                    ap_from_matches(EV[key], identity, 3))
        print("OBSERVED", key, obs[key], flush=True)

    store = {key: (np.zeros(args.n_boot), np.zeros(args.n_boot), np.zeros(args.n_boot)) for key in keys}
    with Pool(args.workers, initializer=_init, initargs=(EV, draws, keys)) as pool:
        for done, (idx, out) in enumerate(pool.imap_unordered(_work, range(args.n_boot)), 1):
            for key in keys:
                for m in range(3):
                    store[key][m][idx] = out[key][m]
            if done % 10 == 0:
                print("BOOT_PROGRESS", done, "%.0fs" % (time.time() - t0), flush=True)

    tag = args.tag or args.model
    payload = {"model": args.model, "source": args.src, "arms": arms, "versions": list(versions),
               "n_boot": int(args.n_boot), "seed": BOOT_SEED, "draws_file": args.draws,
               "observed": {("%s|%s|%d" % (k[0], k[1], mi)): float(obs[k][mi]) for k in keys for mi in range(3)}}
    np.savez_compressed(os.path.join(args.out_dir, "task_y_boot_%s.npz" % tag),
                        {("%s|%s|%d" % (k[0], k[1], m)): store[k][m] for k in keys for m in range(3)})
    json.dump(payload, open(os.path.join(args.out_dir, "task_y_boot_%s_meta.json" % tag), "w"),
              indent=1, sort_keys=True)
    print("DONE", tag, "%.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
