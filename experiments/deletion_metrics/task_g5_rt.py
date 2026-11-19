"""Task G5 on RT-DETRv2-R50vd: the G4 whole-image attention-dose deletion rule.

Dose weights: for the matched query, each (head, sampling point) post-softmax weight divided by
the number of heads, accumulated into the 16-pixel patch that contains its sampling location
(all feature levels), then normalised so that the whole-image sum is 1. Patches are deleted over
the whole image in descending saliency order; step k deletes until the cumulative dose reaches
0.02k, with the crossing patch blended at pixel level.
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

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.models.rt_detr_v2 import modeling_rt_detr_v2 as M

LOCAL = '_rel(DATA_ROOT, "")/rtdetrv2_r50vd'
COCO_DIR = '_rel(DATA_ROOT, "")/coco'
ANN = COCO_DIR + '/annotations/instances_val2017.json'
SEED_TOKEN = 20270930
PATCH = 16
STEPS = 50
FIXED_R = 4


def patched_forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, position_embeddings=None, reference_points=None,
                    spatial_shapes=None, spatial_shapes_list=None, level_start_index=None,
                    output_attentions=False):
    if position_embeddings is not None:
        hidden_states = hidden_states + position_embeddings
    bs, nq, _ = hidden_states.shape
    seq = encoder_hidden_states.shape[1]
    value = self.value_proj(encoder_hidden_states)
    if attention_mask is not None:
        value = value.masked_fill(~attention_mask[..., None], float(0))
    value = value.view(bs, seq, self.n_heads, self.d_model // self.n_heads)
    sampling_offsets = self.sampling_offsets(hidden_states).view(
        bs, nq, self.n_heads, self.n_levels * self.n_points, 2)
    logits = self.attention_weights(hidden_states).view(
        bs, nq, self.n_heads, self.n_levels * self.n_points)
    weights = F.softmax(logits, -1)
    n_points_scale = self.n_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
    offset = sampling_offsets * n_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
    locations = reference_points[:, :, None, :, :2] + offset
    cap = getattr(self, '_rt_capture', None)
    if cap is not None:
        cap['w'] = weights.detach()
        cap['loc'] = locations.detach()
    out = M.multi_scale_deformable_attention_v2(
        value, spatial_shapes_list, locations, weights, self.n_points_list, self.method)
    out = self.output_proj(out)
    return out, weights


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def area_bin(area):
    return 'small' if area < 32 ** 2 else ('medium' if area < 96 ** 2 else 'large')


def giou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    bb = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    union = aa + bb - inter
    iou = inter / max(union, 1e-12)
    cw = max(a[2], b[2]) - min(a[0], b[0]); ch = max(a[3], b[3]) - min(a[1], b[1])
    enc = max(cw, 1e-12) * max(ch, 1e-12)
    return iou - (enc - union) / enc


def match(probs, boxes_px, gt_labels, gt_boxes, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
    n_g = len(gt_boxes)
    if n_g == 0:
        return []
    n_q = probs.shape[0]
    c_class = -probs[:, gt_labels].detach().cpu().numpy()
    c_l1 = np.abs(boxes_px[:, None, :] - gt_boxes[None, :, :]).sum(-1) / 640.0
    c_giou = np.array([[1.0 - giou(boxes_px[i], gt_boxes[j]) for j in range(n_g)] for i in range(n_q)])
    cost = cost_class * c_class + cost_bbox * c_l1 + cost_giou * c_giou
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def patch_grid(h, w):
    return (h + PATCH - 1) // PATCH, (w + PATCH - 1) // PATCH


def upsample_alpha(alpha_patch, h, w):
    a = np.repeat(np.repeat(alpha_patch, PATCH, axis=0), PATCH, axis=1)
    return a[:h, :w]


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


def deletion_sequence(saliency, amounts):
    order = np.argsort(-saliency.ravel(), kind='stable')
    total = len(order)
    seq = []
    for a in amounts:
        a = min(float(a), float(total))
        full = int(math.floor(a + 1e-9))
        frac = a - full
        alpha = np.zeros(total, dtype=np.float32)
        if full:
            alpha[order[:full]] = 1.0
        if frac > 1e-9 and full < total:
            alpha[order[full]] = float(frac)
        seq.append(alpha.reshape(saliency.shape))
    return seq


def apply_alpha(tensor, alpha_np):
    h, w = tensor.shape[-2:]
    a = torch.from_numpy(upsample_alpha(alpha_np, h, w)).to(tensor.device).unsqueeze(0).unsqueeze(0)
    return tensor * (1.0 - a)


def boxes_px(boxes, w, h):
    return torch.stack([((boxes[:, 0] - boxes[:, 2] / 2) * w),
                        ((boxes[:, 1] - boxes[:, 3] / 2) * h),
                        ((boxes[:, 0] + boxes[:, 2] / 2) * w),
                        ((boxes[:, 1] + boxes[:, 3] / 2) * h)], -1)


def target_score(logits, boxes, cat_idx, gt_box, size):
    w, h = size
    probs = logits[0].sigmoid()
    b = boxes_px(boxes[0], w, h)
    p = probs[:, cat_idx]
    x1 = torch.maximum(gt_box[0], b[:, 0]); y1 = torch.maximum(gt_box[1], b[:, 1])
    x2 = torch.minimum(gt_box[2], b[:, 2]); y2 = torch.minimum(gt_box[3], b[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    aa = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    bb = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    iou = inter / (aa + bb - inter).clamp(min=1e-12)
    return float((p * iou).max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--tag', default='g_dev_rt')
    ap.add_argument('--out-dir', default='records/deletion_metrics')
    ap.add_argument('--chunk', type=int, default=10)
    ap.add_argument('--audit-only', action='store_true')
    ap.add_argument('--data', choices=['coco', 'voc'], default='coco')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--voc-json', default='_rel(DATA_ROOT, "")/voc2007test/instances_voc2007test.json')
    ap.add_argument('--voc-images', default='_rel(DATA_ROOT, "")/VOCdevkit/VOC2007/JPEGImages')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    args = ap.parse_args()

    proc = AutoImageProcessor.from_pretrained(LOCAL)
    model = AutoModelForObjectDetection.from_pretrained(LOCAL, torch_dtype=torch.float32).cuda().eval()
    weights_sha = sha256_file(LOCAL + '/model.safetensors')
    mean = list(proc.image_mean); std = list(proc.image_std)
    tf = T.Compose([T.Resize((640, 640)), T.ToTensor(), T.Normalize(mean, std)])

    M.RTDetrV2MultiscaleDeformableAttention.forward = patched_forward
    last = model.model.decoder.layers[-1].encoder_attn
    cap = {}

    cats_order = [c['id'] for c in json.load(open(ANN))['categories']]
    cat2idx = {cid: i for i, cid in enumerate(cats_order)}
    if args.data == 'voc':
        coco = COCO(args.voc_json)
        all_ids = sorted(int(x) for x in coco.getImgIds())
        base_ids = all_ids[args.offset:] if not args.limit else all_ids[args.offset:args.offset + args.limit]
        VOC_IDS = (1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72)
        voc_idx_t = torch.tensor(sorted(cat2idx[c] for c in VOC_IDS), device='cuda', dtype=torch.long)
    else:
        coco = COCO(ANN)
        perm = np.random.Generator(np.random.PCG64(20270925)).permutation(sorted(coco.getImgIds()))
        pool = [int(x) for x in perm[1000:5000]]
        base_ids = pool[:args.limit] if args.limit else pool
    _n = len(base_ids)
    _per = (_n + args.shard_count - 1) // args.shard_count
    image_ids = base_ids[args.shard_index * _per:(args.shard_index + 1) * _per]
    print('G5RT_SHARD', json.dumps({'shard_index': args.shard_index, 'shard_count': args.shard_count,
                                    'base_n': _n, 'n_images': len(image_ids),
                                    'first_image_id': image_ids[0] if image_ids else None,
                                    'last_image_id': image_ids[-1] if image_ids else None},
                                   sort_keys=True), flush=True)

    out_path = os.path.join(args.out_dir, 'task_%s.jsonl.gz' % args.tag)
    rows = []
    checks = {'full_delete': []}
    cnt = collections.Counter()
    t0 = time.time()

    with gzip.open(out_path, 'wt') as fh, torch.no_grad():
        for ordinal, image_id in enumerate(image_ids, 1):
            info = coco.loadImgs([image_id])[0]
            if args.data == 'voc':
                img_path = os.path.join(args.voc_images, info['file_name'])
            else:
                img_path = os.path.join(COCO_DIR, 'val2017', info['file_name'])
            image = Image.open(img_path).convert('RGB')
            size = (int(info['width']), int(info['height']))
            x = tf(image)
            _, hgt, wgt = x.shape
            batch = x.unsqueeze(0).cuda()

            last._rt_capture = cap
            out = model(pixel_values=batch)
            last._rt_capture = None
            attn = cap['w'][0].clone()    # (nq, heads, levels*points)
            loc_q = cap['loc'][0].clone()  # (nq, heads, levels*points, 2)
            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))]
            if not anns:
                cnt['images_no_anns'] += 1
                continue
            gt_labels = [cat2idx[a['category_id']] for a in anns]
            gt_boxes = np.array([[a['bbox'][0], a['bbox'][1],
                                  a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns])
            probs = out.logits[0].sigmoid()
            box_px = boxes_px(out.pred_boxes[0], size[0], size[1]).cpu().numpy()
            if args.data == 'voc':
                keep = torch.nonzero(torch.isin(probs.argmax(-1), voc_idx_t), as_tuple=False).flatten()
                keep_np = keep.cpu().numpy()
                sub = match(probs[keep], box_px[keep_np], gt_labels, gt_boxes)
                pairs = [(int(keep_np[q]), int(g)) for q, g in sub]
            else:
                pairs = match(probs, box_px, gt_labels, gt_boxes)
            best = {}
            for q, g in pairs:
                ann = anns[g]
                b = area_bin(float(ann['bbox'][2]) * float(ann['bbox'][3]))
                if b not in best or ann['id'] < best[b][2]['id']:
                    best[b] = (q, g, ann)
            for _b in ('small', 'medium', 'large'):
                if _b not in best:
                    cnt['no_match_' + _b] += 1

            gh, gw = patch_grid(hgt, wgt)
            # patch grid is in the resized-input frame (640x640); GT boxes are original pixels
            sx = wgt / float(size[0])
            sy = hgt / float(size[1])
            yy, xx = np.indices((gh, gw))
            xs0 = xx.ravel() * PATCH
            ys0 = yy.ravel() * PATCH

            for bin_name, (q, g, ann) in best.items():
                bx, by, bw, bh = [float(v) for v in ann['bbox']]
                gt_box = torch.tensor([bx, by, bx + bw, by + bh], device='cuda', dtype=torch.float32)
                cat_idx = cat2idx[ann['category_id']]
                with torch.no_grad():
                    clean_score = target_score(out.logits, out.pred_boxes, cat_idx, gt_box, size)
                cnt['targets_' + bin_name] += 1
                if clean_score <= 0:
                    cnt['excluded_zero_score_' + bin_name] += 1
                    continue
                if args.audit_only:
                    cnt['audited_' + bin_name] += 1
                    continue

                wq = attn[q].mean(0)  # (P,)
                loc = loc_q[q].mean(0)
                if loc is None:
                    sal_attn = np.zeros((gh, gw), dtype=np.float64)
                else:
                    gx = (loc[:, 0] * gw).clamp(0, gw - 1).long().cpu().numpy()
                    gy = (loc[:, 1] * gh).clamp(0, gh - 1).long().cpu().numpy()
                    sal_attn = np.zeros((gh, gw), dtype=np.float64)
                    np.add.at(sal_attn, (gy, gx), wq.cpu().numpy())

                rng = np.random.Generator(np.random.PCG64(
                    SEED_TOKEN * 1000003 + int(image_id) * 1009 + int(ann['id'])))
                sal_rand = rng.random((gh, gw))

                t_in = x.clone().cuda().requires_grad_(True)
                with torch.enable_grad():
                    out_g = model(pixel_values=t_in.unsqueeze(0))
                    prob = out_g.logits[0][q].sigmoid()[cat_idx]
                    grad = torch.autograd.grad(prob, t_in)[0]
                gmag = grad.abs().sum(0).float().cpu().numpy()
                pad_h, pad_w = gh * PATCH - hgt, gw * PATCH - wgt
                gp = np.pad(gmag, ((0, pad_h), (0, pad_w)))
                sal_grad = gp.reshape(gh, PATCH, gw, PATCH).sum(axis=(1, 3))

                bx1, by1 = bx * sx, by * sy
                bx2, by2 = (bx + bw) * sx, (by + bh) * sy
                in_box = ((xs0 + PATCH / 2 >= bx1) & (xs0 + PATCH / 2 <= bx2) &
                          (ys0 + PATCH / 2 >= by1) & (ys0 + PATCH / 2 <= by2)).reshape(gh, gw)
                n_box = int(in_box.sum())

                n_heads = int(attn.shape[1])
                dose_img = np.zeros((gh, gw), dtype=np.float64)
                ww = (attn[q] / n_heads).cpu().numpy().ravel()
                gxd = (loc_q[q][..., 0] * gw).clamp(0, gw - 1).long().cpu().numpy().ravel()
                gyd = (loc_q[q][..., 1] * gh).clamp(0, gh - 1).long().cpu().numpy().ravel()
                np.add.at(dose_img, (gyd, gxd), ww)
                _tot = float(dose_img.sum())
                if _tot <= 1e-9:
                    cnt['excluded_zero_image_dose_' + bin_name] += 1
                    continue
                dose_patch = dose_img / _tot
                dose_in = dose_patch * in_box
                raw_mass = float(dose_in.sum())

                for sal_name, sal in (('random', sal_rand), ('attn', sal_attn), ('grad', sal_grad)):
                    for axis in ('dose_full',):
                        budgets = [0.02 * (t + 1) for t in range(STEPS)]
                        seq, removed, total_dose = dose_sequence_full(sal, dose_patch, budgets)
                        scores = [clean_score]
                        for i in range(0, len(seq), args.chunk):
                            ch = seq[i:i + args.chunk]
                            b = torch.cat([apply_alpha(x, a) for a in ch], 0).cuda()
                            o = model(pixel_values=b)
                            for j in range(b.shape[0]):
                                scores.append(target_score(o.logits[j:j + 1], o.pred_boxes[j:j + 1],
                                                           cat_idx, gt_box, size))
                        norm = [s / clean_score for s in scores]
                        box_frac = [float((a * in_box).sum()) / max(n_box, 1) for a in seq]
                        box_dose = ([float((a * dose_in).sum()) / raw_mass if raw_mass > 0 else 0.0
                                     for a in seq])
                        row = {'image_id': int(image_id), 'annotation_id': int(ann['id']),
                               'bin': bin_name, 'saliency': sal_name, 'axis': axis,
                               'clean_score': clean_score, 'n_box_patches': n_box,
                               'image_dose_total': total_dose,
                               'dose_removed': [0.0] + removed,
                               'steps': norm, 'auc': float(np.mean(norm[1:])),
                               'box_removed_frac': [0.0] + box_frac,
                               'box_dose_frac': [0.0] + box_dose}
                        rows.append(row)
                        fh.write(json.dumps(row, sort_keys=True) + '\n')

                full = model(pixel_values=apply_alpha(x, np.ones((gh, gw), dtype=np.float32)).cuda())
                fs = target_score(full.logits, full.pred_boxes, cat_idx, gt_box, size)
                rec = {'image_id': int(image_id), 'annotation_id': int(ann['id']), 'bin': bin_name,
                       'kind': 'full_delete', 'clean_score': clean_score,
                       'full_delete_score': fs, 'ratio': fs / clean_score if clean_score else 0.0}
                checks['full_delete'].append(rec)
                fh.write(json.dumps(rec, sort_keys=True) + '\n')

            if ordinal % 10 == 0:
                print('G_RT_PROGRESS', json.dumps({'n': ordinal, 'elapsed_s': round(time.time() - t0, 1)},
                                                  sort_keys=True), flush=True)

    zero_maxabs = max([abs(r['steps'][0] - 1.0) for r in rows], default=0.0)
    ratios = [c['ratio'] for c in checks['full_delete']]
    proc_out = {'model': 'rtdetr_v2_r50vd', 'tag': args.tag, 'n_images': len(image_ids),
                'n_rows': len(rows), 'elapsed_s': round(time.time() - t0, 1),
                'jsonl_sha256': sha256_file(out_path), 'path': out_path,
                'weights_sha256': weights_sha, 'zero_step_maxabs': zero_maxabs,
                'n_full_delete': len(ratios),
                'full_delete_mean_ratio': float(np.mean(ratios)) if ratios else None,
                'full_delete_max_ratio': float(np.max(ratios)) if ratios else None}
    proc_out['counts'] = dict(cnt)
    print('G_RT_DONE', json.dumps(proc_out, sort_keys=True), flush=True)
    open(os.path.join(args.out_dir, 'task_%s_meta.json' % args.tag), 'w').write(
        json.dumps(proc_out, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
