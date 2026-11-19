"""Task G5: the Task G4 deletion rule (whole-image attention-dose step) on COCO or VOC.

Identical to task_g4.py except (i) a --data coco|voc branch and (ii) box_removed_frac is
now summed over the in-box mask only. The G4 dev run of record stays reproducible through
task_g4.py, whose SHA256 is cited in RESULTS_G4_dev.md; this file is the frozen G5 code.

Only in-box patches are deleted; step k removes patches in descending saliency
order over ALL patches of the image until the cumulative whole-image attention
dose reaches k*0.02. The crossing patch is blended by the dose fraction it
contributes. In-box deleted patch/dose fractions are recorded per step.
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
import collections
import gzip
import hashlib
import json
import math
import os
import sys
import time

_CODE = '_rel(REPO_ROOT, "")' if os.path.isdir('_rel(REPO_ROOT, "")') else '_rel(REPO_ROOT, "")'
sys.path.insert(0, _CODE)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

from intervene import Controller, ground_truth
from mve import load_model

SEED_TOKEN = 20270930
BOOT_SEED = 20271001
PATCH = 16
STEPS = 50
FIXED_R = 4
VOC_COCO_IDS = (1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21,
                44, 62, 63, 64, 67, 72)  # 20 VOC classes as COCO category ids
N_BOOT = 2000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def area_bin(area):
    if area < 32 ** 2:
        return 'small'
    if area < 96 ** 2:
        return 'medium'
    return 'large'


def patch_grid(h, w):
    return (h + PATCH - 1) // PATCH, (w + PATCH - 1) // PATCH


def upsample_alpha(alpha_patch, h, w):
    """(gh,gw) alpha -> (h,w) with exact 16-px blocks."""
    a = np.repeat(np.repeat(alpha_patch, PATCH, axis=0), PATCH, axis=1)
    return a[:h, :w]


def deletion_sequence(saliency, amounts, h, w, allowed=None):
    """Return list of (gh,gw) alpha arrays, one per step, for cumulative amounts.

    With `allowed`, only patches inside it can be deleted and the ranking is
    restricted to them (Task G2 in-box relative rule)."""
    flat = saliency.ravel()
    if allowed is None:
        order = np.argsort(-flat, kind='stable')
    else:
        idx = np.flatnonzero(allowed.ravel())
        order = idx[np.argsort(-flat[idx], kind='stable')]
    total = len(order)
    seq = []
    for a in amounts:
        a = min(float(a), float(total))
        full = int(math.floor(a + 1e-9))
        frac = a - full
        alpha = np.zeros(flat.size, dtype=np.float32)
        if full:
            alpha[order[:full]] = 1.0
        if frac > 1e-9 and full < total:
            alpha[order[full]] = float(frac)
        seq.append(alpha.reshape(saliency.shape))
    return seq


def dose_sequence_full(saliency, dose_patch, budgets):
    """Deletion sequence whose steps are defined by cumulative whole-image dose."""
    flat = saliency.ravel()
    dose = dose_patch.ravel()
    order = np.argsort(-flat, kind='stable')
    cumulative = np.cumsum(dose[order])
    total = float(cumulative[-1]) if len(order) else 0.0
    seq, removed = [], []
    for budget in budgets:
        target = min(float(budget), total)
        alpha = np.zeros(flat.size, dtype=np.float32)
        if total > 0:
            crossing = int(np.searchsorted(cumulative, target - 1e-12, side='left'))
            crossing = min(crossing, len(order) - 1)
            before = float(cumulative[crossing - 1]) if crossing > 0 else 0.0
            if crossing > 0:
                alpha[order[:crossing]] = 1.0
            patch_dose = float(dose[order[crossing]])
            fraction = 0.0 if patch_dose <= 0 else (target - before) / patch_dose
            fraction = float(min(max(fraction, 0.0), 1.0))
            if fraction > 0:
                alpha[order[crossing]] = fraction
        seq.append(alpha.reshape(saliency.shape))
        removed.append(target)
    return seq, removed, total


def apply_alpha(tensor, alpha_np):
    h, w = tensor.shape[-2:]
    a = torch.from_numpy(upsample_alpha(alpha_np, h, w)).to(tensor.device).unsqueeze(0).unsqueeze(0)
    return tensor * (1.0 - a)


def raw_dets(output, model_name):
    logits = output['pred_logits'][0]
    boxes = output['pred_boxes'][0]
    if model_name == 'conditional_detr':
        probs = logits.sigmoid()
        scores, labels = probs.max(-1)
    else:
        probs = logits.softmax(-1)[..., :-1]
        scores, labels = probs.max(-1)
    return scores, labels, boxes


def boxes_xyxy(boxes, w, h):
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack(((cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h), -1)


def iou_one_to_many(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0]); y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2]); y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    aa = (box[2] - box[0]) * (box[3] - box[1])
    bb = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (aa + bb - inter).clamp(min=1e-12)


def target_score(output, model_name, c_g, gt_box, size):
    """max over same-class predictions of p * IoU, in image coordinates."""
    w, h = size
    scores, labels, boxes = raw_dets(output, model_name)
    sel = labels == c_g
    if not bool(sel.any()):
        return 0.0
    b = boxes_xyxy(boxes[sel], w, h)
    return float((scores[sel] * iou_one_to_many(gt_box, b)).max().item())


def auc_trapz(values):
    v = np.asarray(values, dtype=np.float64)
    return float(v.sum() / max(len(v), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='_rel(DATA_ROOT, "")')
    ap.add_argument('--model', choices=['detr', 'conditional_detr'], default='detr')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--tag', default='g_dev_detr')
    ap.add_argument('--out-dir', default='_rel(OUT_ROOT, "")')
    ap.add_argument('--axes', default='dose_full')
    ap.add_argument('--chunk', type=int, default=10)
    ap.add_argument('--bootstrap', type=int, default=N_BOOT)
    ap.add_argument('--audit-only', action='store_true')
    ap.add_argument('--data', choices=['coco', 'voc'], default='coco')
    ap.add_argument('--voc-json', default='_rel(DATA_ROOT, "")/voc2007test/instances_voc2007test.json')
    ap.add_argument('--voc-images', default='_rel(DATA_ROOT, "")/VOCdevkit/VOC2007/JPEGImages')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    model, post, _module, transform = load_model(args.model, args.cache)
    from util.misc import NestedTensor
    last_layer = model.transformer.decoder.layers[-1]
    if hasattr(last_layer, 'multihead_attn'):
        attn_module = last_layer.multihead_attn
    elif hasattr(last_layer, 'cross_attn'):
        attn_module = last_layer.cross_attn
    else:
        raise AttributeError('no cross-attention module on last decoder layer')
    controller = Controller(attn_module)

    matcher = __import__('models.matcher', fromlist=['HungarianMatcher'])
    HungarianMatcher = matcher.HungarianMatcher
    try:
        hm = HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                              use_focal_loss=True, alpha=0.25, gamma=2.0)
    except TypeError:
        hm = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

    if args.data == 'voc':
        coco = COCO(args.voc_json)
        all_ids = sorted(int(x) for x in coco.getImgIds())
        base_ids = all_ids[args.offset:] if not args.limit else all_ids[args.offset:args.offset + args.limit]
        voc_ids = torch.tensor(VOC_COCO_IDS, device='cuda', dtype=torch.long)
    else:
        coco = COCO(os.path.join(args.cache, 'coco/annotations/instances_val2017.json'))
        perm = np.random.Generator(np.random.PCG64(20270925)).permutation(sorted(coco.getImgIds()))
        holdout = [int(x) for x in perm[1000:5000]]
        base_ids = holdout[args.offset:args.offset + args.limit]
        voc_ids = None
    _n = len(base_ids)
    _per = (_n + args.shard_count - 1) // args.shard_count
    image_ids = base_ids[args.shard_index * _per:(args.shard_index + 1) * _per]
    print('G5_SHARD', json.dumps({'shard_index': args.shard_index, 'shard_count': args.shard_count,
                                  'base_n': _n, 'n_images': len(image_ids),
                                  'first_image_id': image_ids[0] if image_ids else None,
                                  'last_image_id': image_ids[-1] if image_ids else None},
                                 sort_keys=True), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'task_%s.jsonl.gz' % args.tag)
    t0 = time.time()
    rows = []
    checks = {'zero_step_maxabs': 0.0, 'full_delete': []}
    cnt = collections.Counter()

    with gzip.open(out_path, 'wt') as fh:
        for ordinal, image_id in enumerate(image_ids, 1):
            info = coco.loadImgs([image_id])[0]
            img_path = (os.path.join(args.voc_images, info['file_name']) if args.data == 'voc'
                        else os.path.join(args.cache, 'coco/val2017', info['file_name']))
            image = Image.open(img_path).convert('RGB')
            tensor, _ = transform(image, None)
            tensor = tensor.unsqueeze(0).cuda()
            _, _, hgt, wgt = tensor.shape
            size = (int(info['width']), int(info['height']))

            with torch.no_grad():
                out = controller.forward(model, tensor[0])
            attn = controller.weights  # (nq, ns) head-mean, last layer
            anns, targets = ground_truth(coco, image_id, size[0], size[1])
            if len(anns) == 0:
                cnt['images_no_anns'] += 1
                continue
            if args.data == 'voc':
                _sc, labels_all, _bx = raw_dets(out, args.model)
                keep = torch.nonzero(torch.isin(labels_all, voc_ids), as_tuple=False).flatten()
                sub_out = {'pred_logits': out['pred_logits'][:, keep],
                           'pred_boxes': out['pred_boxes'][:, keep]}
                with torch.no_grad():
                    idx = hm(sub_out, [targets])[0]
                pairs = [(int(keep[q]), int(g)) for q, g in zip(idx[0].tolist(), idx[1].tolist())]
            else:
                with torch.no_grad():
                    idx = hm(out, [targets])[0]
                pairs = [(int(q), int(g)) for q, g in zip(idx[0].tolist(), idx[1].tolist())]
            best = {}
            for q, g in pairs:
                ann = anns[g]
                b = area_bin(float(ann['bbox'][2]) * float(ann['bbox'][3]))
                if b not in best or ann['id'] < best[b][2]['id']:
                    best[b] = (q, g, ann)
            for _b in ('small', 'medium', 'large'):
                if _b not in best:
                    cnt['no_match_' + _b] += 1

            fh_map = (hgt + 31) // 32
            fw_map = (wgt + 31) // 32
            assert fh_map * fw_map == attn.shape[1], (fh_map, fw_map, attn.shape)
            gh, gw = patch_grid(hgt, wgt)
            # patch grid lives in the resized-input frame; GT boxes are in original-image pixels
            sx = wgt / float(size[0])
            sy = hgt / float(size[1])
            yy, xx = np.indices((gh, gw))
            xs0 = xx.ravel() * PATCH
            ys0 = yy.ravel() * PATCH

            for bin_name, (q, g, ann) in best.items():
                x, y, bw, bh = [float(v) for v in ann['bbox']]
                gt_box = torch.tensor([[x, y, x + bw, y + bh]], device='cuda', dtype=torch.float32)[0]
                c_g = int(ann['category_id'])
                with torch.no_grad():
                    clean_out = out
                    clean_score = target_score(clean_out, args.model, c_g, gt_box, size)
                cnt['targets_' + bin_name] += 1
                if clean_score <= 0:
                    cnt['excluded_zero_score_' + bin_name] += 1
                    continue
                if args.audit_only:
                    cnt['audited_' + bin_name] += 1
                    continue

                a_map = torch.from_numpy(attn[q].reshape(fh_map, fw_map)).float()[None, None]
                a_up = F.interpolate(a_map, size=(gh, gw), mode='bilinear', align_corners=False)
                sal_attn = a_up[0, 0].numpy().astype(np.float64)

                rng = np.random.Generator(np.random.PCG64(
                    SEED_TOKEN * 1000003 + int(image_id) * 1009 + int(ann['id'])))
                sal_rand = rng.random((gh, gw))

                t_in = tensor[0].detach().clone().requires_grad_(True)
                nt = NestedTensor(t_in.unsqueeze(0),
                                  torch.zeros((1, hgt, wgt), dtype=torch.bool, device=t_in.device))
                with torch.enable_grad():
                    out_g = model(nt)
                logits = out_g['pred_logits'][0]
                if args.model == 'conditional_detr':
                    prob = logits[q].sigmoid()[c_g]
                else:
                    prob = logits[q].softmax(-1)[c_g]
                grad = torch.autograd.grad(prob, t_in, retain_graph=False)[0]
                gmag = grad.abs().sum(0).detach().float().cpu().numpy()
                gh2, gw2 = patch_grid(hgt, wgt)
                pad_h, pad_w = gh2 * PATCH - hgt, gw2 * PATCH - wgt
                gp = np.pad(gmag, ((0, pad_h), (0, pad_w)))
                sal_grad = gp.reshape(gh2, PATCH, gw2, PATCH).sum(axis=(1, 3))

                bx1, by1 = x * sx, y * sy
                bx2, by2 = (x + bw) * sx, (y + bh) * sy
                in_box = ((xs0 + PATCH / 2 >= bx1) & (xs0 + PATCH / 2 <= bx2) &
                          (ys0 + PATCH / 2 >= by1) & (ys0 + PATCH / 2 <= by2)).reshape(gh, gw)
                n_box = int(in_box.sum())

                dose_in = sal_attn * in_box
                raw_mass = float(dose_in.sum())
                if raw_mass <= 1e-9:
                    cnt['excluded_zero_dose_' + bin_name] += 1
                    continue
                raw_mass_img = float(sal_attn.sum())
                if raw_mass_img <= 1e-9:
                    cnt['excluded_zero_image_dose_' + bin_name] += 1
                    continue
                dose_patch = sal_attn / raw_mass_img
                dose_patch_in = dose_in / raw_mass

                for sal_name, sal in (('random', sal_rand), ('attn', sal_attn), ('grad', sal_grad)):
                    for axis in args.axes.split(','):
                        if axis == 'dose_full':
                            budgets = [0.02 * (t + 1) for t in range(STEPS)]
                            seq, removed_dose, total_dose = dose_sequence_full(sal, dose_patch, budgets)
                            amounts = None
                        else:
                            raise ValueError('unknown axis: ' + axis)
                        scores = [clean_score]
                        chunks = [seq[i:i + args.chunk] for i in range(0, len(seq), args.chunk)]
                        with torch.no_grad():
                            for ch in chunks:
                                batch = torch.cat([apply_alpha(tensor[0], a) for a in ch], 0)
                                o = model(list(batch))
                                for i in range(batch.shape[0]):
                                    single = {'pred_logits': o['pred_logits'][i:i + 1],
                                              'pred_boxes': o['pred_boxes'][i:i + 1]}
                                    scores.append(target_score(single, args.model, c_g, gt_box, size))
                        norm = [s / clean_score for s in scores]
                        box_frac = [float((a * in_box).sum()) / max(n_box, 1) for a in seq]
                        box_frac = [0.0] + box_frac
                        box_dose = [float((a * dose_patch_in).sum()) for a in seq]
                        box_dose = [0.0] + box_dose
                        row = {'image_id': int(image_id), 'annotation_id': int(ann['id']),
                               'bin': bin_name, 'saliency': sal_name, 'axis': axis,
                               'clean_score': clean_score, 'n_box_patches': n_box,
                               'image_dose_total': total_dose,
                               'dose_removed': [0.0] + removed_dose,
                               'steps': norm, 'auc': auc_trapz(norm[1:]),
                               'box_removed_frac': box_frac,
                               'box_dose_frac': box_dose}
                        rows.append(row)
                        fh.write(json.dumps(row, sort_keys=True) + '\n')

                with torch.no_grad():
                    full = model([apply_alpha(tensor[0], np.ones((gh, gw), dtype=np.float32))[0]])
                full_score = target_score(full, args.model, c_g, gt_box, size)
                row_check = {'image_id': int(image_id), 'annotation_id': int(ann['id']),
                             'bin': bin_name, 'kind': 'full_delete',
                             'clean_score': clean_score, 'full_delete_score': full_score,
                             'ratio': (full_score / clean_score) if clean_score else 0.0}
                checks['full_delete'].append(row_check)
                fh.write(json.dumps(row_check, sort_keys=True) + '\n')

            logger = {'n_images': ordinal, 'elapsed_s': round(time.time() - t0, 1)}
            if ordinal % 10 == 0:
                print('G_PROGRESS', json.dumps(logger, sort_keys=True), flush=True)

    zero_maxabs = max([abs(r['steps'][0] - 1.0) for r in rows], default=0.0)
    full_ratios = [c['ratio'] for c in checks['full_delete']]
    proc = {'model': args.model, 'tag': args.tag, 'n_images': len(image_ids),
            'n_rows': len(rows), 'elapsed_s': round(time.time() - t0, 1),
            'jsonl_sha256': sha256_file(out_path), 'path': out_path,
            'zero_step_maxabs': zero_maxabs,
            'n_full_delete': len(full_ratios),
            'full_delete_mean_ratio': float(np.mean(full_ratios)) if full_ratios else None,
            'full_delete_max_ratio': float(np.max(full_ratios)) if full_ratios else None}
    proc['counts'] = dict(cnt)
    print('G_DONE', json.dumps(proc, sort_keys=True), flush=True)
    open(os.path.join(args.out_dir, 'task_%s_meta.json' % args.tag), 'w').write(
        json.dumps(proc, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
