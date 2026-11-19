"""N1-N3 for Task N on VOC 2007 test.

The registered statistic is Delta_lesion = E[Q_clean - Q_lesion | small]
- E[Q_clean - Q_lesion | large]. The run script stores D4 (which uses the
outside control) but not L for the four-pair arm, so L is computed from the
per-anchor rows for every arm used in a hypothesis.
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
import argparse
import gzip
import json

import numpy as np

NB = 5000
SEED = 20271001


def rows_of(path):
    out = []
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def cell_values(rows, arm, bin_name, spec):
    sel = []
    for r in rows:
        if r['arm'] != arm or r['area_bin'] != bin_name:
            continue
        # The supplementary spec only exists for the region arms; a four-pair
        # lesion cannot cover a whole head.
        if spec == 'supp' and arm.startswith('region'):
            if int(r.get('head_zero_heads', 0)) > 0 or int(r.get('region_pairs', 0)) == 0:
                continue
        q0 = float(r['clean_local']['quality'])
        q1 = float(r['perturbed_local']['quality'])
        if not np.isfinite(q0 - q1):
            continue
        sel.append((int(r['image_id']), float(r['removed_attention_mass']), q0 - q1))
    return sel


def image_vector(idx, sel):
    """Per-image mean of the selected rows, NaN where the image has none.

    This reproduces stage7_analysis.image_vector, the estimator frozen with the
    development runs: the bootstrap unit is the image, not the anchor.
    """
    buckets = {}
    for im, _mass, value in sel:
        buckets.setdefault(im, []).append(value)
    vec = np.full(len(idx), np.nan, dtype=float)
    for im, vals in buckets.items():
        vec[idx[im]] = float(np.mean(vals))
    return vec


def contrast(rows, arm, spec, idx):
    out = {}
    for name in ('small', 'large'):
        sel = cell_values(rows, arm, name, spec)
        out[name] = {
            'n': len(sel),
            'mass': float(np.mean([s[1] for s in sel])) if sel else float('nan'),
            'loss': float(np.mean([s[2] for s in sel])) if sel else float('nan'),
            'vec': image_vector(idx, sel),
        }
    a, b = out['small']['vec'], out['large']['vec']
    observed = float(np.nanmean(a) - np.nanmean(b))
    rng = np.random.Generator(np.random.PCG64(SEED))
    boot = []
    n = len(a)
    for _ in range(0, NB, 100):
        draw = rng.integers(0, n, size=(100, n))
        with np.errstate(invalid='ignore'):
            boot.extend((np.nanmean(a[draw], axis=1) - np.nanmean(b[draw], axis=1)).tolist())
    boot = np.array(boot, dtype=float)
    res = {'ratio_mass': out['small']['mass'] / out['large']['mass'],
           'n_small': out['small']['n'], 'n_large': out['large']['n'],
           'n_images_small': int(np.isfinite(a).sum()), 'n_images_large': int(np.isfinite(b).sum()),
           'mass_small': out['small']['mass'], 'mass_large': out['large']['mass'],
           'loss_small': out['small']['loss'], 'loss_large': out['large']['loss'],
           'delta': observed,
           'delta_ci': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
           'delta_draws': boot}
    return res, out


def main():
    ap = argparse.ArgumentParser()
    # Either model may be omitted when its per-anchor record is not shipped.
    ap.add_argument('--deim', default=None)
    ap.add_argument('--rt', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--coco-json', default=None,
                    help='annotation file whose image list defines the bootstrap pool')
    ap.add_argument('--dev-holdout', action='store_true',
                    help='use the first 1,000 images of the frozen COCO holdout as the pool')
    a = ap.parse_args()
    results = {}
    for name, path in (('DEIMv2-S', a.deim), ('RT-DETRv2-R50vd', a.rt)):
        if not path:
            continue
        rows = rows_of(path)
        if a.dev_holdout:
            ann = json.load(open('_rel(DATA_ROOT, "")/coco/annotations/instances_val2017.json'))
            perm = np.random.Generator(np.random.PCG64(20270925)).permutation(
                sorted(int(im['id']) for im in ann['images']))
            # The Task N scripts sort the holdout first and then take the first
            # images, so the pool is sorted(perm[1000:5000])[:1000].
            images = sorted(int(x) for x in perm[1000:5000])[:1000]
        elif a.coco_json:
            ann = json.load(open(a.coco_json))
            images = sorted(int(im['id']) for im in ann['images'])
        else:
            images = sorted({int(r['image_id']) for r in rows})
        idx = {im: k for k, im in enumerate(images)}
        n = len(images)
        entry = {'n_images': n, 'n_rows': len(rows)}
        for spec in ('main', 'supp'):
            r2, _ = contrast(rows, 'region2_inside', spec, idx)
            r4, _ = contrast(rows, 'fixed_inside', spec, idx)
            entry[spec] = {
                'region2': {k: v for k, v in r2.items() if k != 'delta_draws'},
                'four': {k: v for k, v in r4.items() if k != 'delta_draws'},
                'region2_minus_four': [float(x) for x in (r2['delta_draws'] - r4['delta_draws'])],
            }
        results[name] = entry
    json.dump(results, open(a.out, 'w'), indent=1, sort_keys=True)
    for name, e in results.items():
        for spec in ('main', 'supp'):
            r2 = e[spec]['region2']
            r4 = e[spec]['four']
            print('%s [%s] n_img=%d' % (name, spec, e['n_images']))
            print('   N1 mass ratio 2x2      %.4f  (small %.6f n=%d / large %.6f n=%d)' % (
                r2['ratio_mass'], r2['mass_small'], r2['n_small'], r2['mass_large'], r2['n_large']))
            print('   N2 Delta_2x2           %+.6f [%+.6f, %+.6f]' % (
                r2['delta'], r2['delta_ci'][0], r2['delta_ci'][1]))
            print('   N3 Delta_four          %+.6f [%+.6f, %+.6f]' % (
                r4['delta'], r4['delta_ci'][0], r4['delta_ci'][1]))


if __name__ == '__main__':
    main()
