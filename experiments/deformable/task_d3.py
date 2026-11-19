"""Task D3: query-specific lesions on HF Deformable DETR (SenseTime/deformable-detr).

Dev set = first 1,000 COCO holdout images.
  D2 arm set : fixed 4 in-box / 4 out-box pairs, equal mass m in {0.02,0.05,0.10,0.20,0.40}
  N  arm set : fixed 2x2 and 3x3 stride-16-cell square regions, inside and outside control
Unit = (head, sampling point); mass = post-softmax weight / n_heads.
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
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import (DeformableDetrConfig, DeformableDetrForObjectDetection,
                          DeformableDetrImageProcessor)
from transformers.models.deformable_detr import modeling_deformable_detr as HFM

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/code')
sys.path.insert(0, '_rel(REPO_ROOT, "")/vendor/detr')

SELECT_SEED = 20270930
LAYER_INDEX = 2
MASSES = (0.02, 0.05, 0.10, 0.20, 0.40)
SIDES = (2, 3)


def install_patch(cls):
    """Replace attention forward with a version that supports a per-(q,head,pair) bias."""

    def patched_forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                        encoder_attention_mask=None, position_embeddings=None, reference_points=None,
                        spatial_shapes=None, spatial_shapes_list=None, level_start_index=None,
                        output_attentions=False):
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
        batch_size, num_queries, _ = hidden_states.shape
        sequence_length = encoder_hidden_states.shape[1]
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2)
        logits = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points)
        bias = getattr(self, '_d3_bias', None)
        if bias is not None:
            logits = logits + bias
        weights = torch.nan_to_num(F.softmax(logits, -1), nan=0.0).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points)
        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (reference_points[:, :, None, :, None, :]
                                  + sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        elif num_coordinates == 4:
            sampling_locations = (reference_points[:, :, None, :, None, :2]
                                  + sampling_offsets / self.n_points
                                  * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError('reference_points last dim must be 2 or 4')
        capture = getattr(self, '_d3_capture', None)
        if capture is not None:
            capture['weights'] = weights.detach()
            capture['locations'] = sampling_locations.detach()
        output = self.attn(value, spatial_shapes, spatial_shapes_list, level_start_index,
                           sampling_locations, weights, self.im2col_step)
        return self.output_proj(output), weights

    cls.forward = patched_forward


def area_bin(area):
    return 'small' if area < 32 ** 2 else ('medium' if area < 96 ** 2 else 'large')


def take_mass_pairs(order, masses, target):
    cumulative = np.cumsum(masses[order])
    if cumulative[-1] < target - 1e-7:
        return None
    index = int(np.searchsorted(cumulative, target - 1e-10, side='left'))
    index = min(index, len(order) - 1)
    previous = cumulative[index - 1] if index > 0 else 0.0
    fraction = (target - previous) / max(float(masses[order[index]]), 1e-12)
    fraction = float(min(max(fraction, 0.0), 1.0))
    return [(int(order[i]), 1.0) for i in range(index)] + [(int(order[index]), fraction)]


def local_quality(outputs, query, ann, size):
    logits = (outputs['logits'] if 'logits' in outputs else outputs['pred_logits'])[0]
    boxes = outputs['pred_boxes'][0]
    prob = float(torch.sigmoid(logits[query, ann['label_index']]).item())
    cx, cy, bw, bh = boxes[query].tolist()
    width, height = size
    pred = np.array([(cx - bw / 2) * width, (cy - bh / 2) * height,
                     (cx + bw / 2) * width, (cy + bh / 2) * height])
    x, y, w, h = ann['bbox']
    gt = np.array([x, y, x + w, y + h])
    inter = np.maximum(0, np.minimum(pred[2], gt[2]) - np.maximum(pred[0], gt[0])) * \
            np.maximum(0, np.minimum(pred[3], gt[3]) - np.maximum(pred[1], gt[1]))
    union = (pred[2] - pred[0]) * (pred[3] - pred[1]) + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
    iou = float(inter / max(union, 1e-12))
    return {'quality': prob * iou, 'p_gt': prob, 'iou': iou}


def outside_centre(cx, cy, bw, bh, hx, hy, size):
    width, height = size
    step = bw / 2.0 + hx + 4.0
    for dx in (step, -step):
        x2 = cx + dx
        if hx + 1 <= x2 <= width - hx - 1:
            return (x2, cy)
    step = bh / 2.0 + hy + 4.0
    for dy in (step, -step):
        y2 = cy + dy
        if hy + 1 <= y2 <= height - hy - 1:
            return (cx, y2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--identity-images', type=int, default=20)
    ap.add_argument('--data', choices=['coco', 'voc'], default='coco')
    ap.add_argument('--voc-json', default='_rel(DATA_ROOT, "")/voc2007test/instances_voc2007test.json')
    ap.add_argument('--voc-images', default='_rel(DATA_ROOT, "")/VOCdevkit/VOC2007/JPEGImages')
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(SELECT_SEED)
    install_patch(HFM.DeformableDetrMultiscaleDeformableAttention)

    processor = DeformableDetrImageProcessor.from_pretrained(args.dir)
    cfg = DeformableDetrConfig.from_pretrained(args.dir)
    cfg.use_pretrained_backbone = False
    model, info = DeformableDetrForObjectDetection.from_pretrained(
        args.dir, config=cfg, output_loading_info=True)
    model = model.cuda().eval()
    print('D3_LOADINFO', json.dumps({'missing': len(info['missing_keys']),
                                     'unexpected': len(info['unexpected_keys'])}), flush=True)

    attn = model.model.decoder.layers[LAYER_INDEX].encoder_attn
    n_heads = attn.n_heads
    n_levels = attn.n_levels
    n_points = attn.n_points
    n_pairs = n_levels * n_points
    print('D3_ATTENTION_GEOMETRY', json.dumps({'layer': LAYER_INDEX, 'heads': n_heads,
                                               'levels': n_levels, 'points': n_points,
                                               'pairs': n_pairs}), flush=True)

    if args.data == 'voc':
        coco = COCO(args.voc_json)
        image_ids = sorted(int(x) for x in coco.getImgIds())
        image_root = pathlib.Path(args.voc_images)
    else:
        coco = COCO(str(pathlib.Path(args.cache) / 'coco/annotations/instances_val2017.json'))
        perm = np.random.Generator(np.random.PCG64(20270925)).permutation(sorted(coco.getImgIds()))
        image_ids = sorted(int(x) for x in perm[1000:5000])
        image_root = pathlib.Path(args.cache) / 'coco/val2017'
    if args.limit:
        image_ids = image_ids[:args.limit]
    print('D3_DATA', args.data, 'N', len(image_ids), flush=True)
    print('D3_DEV_IMAGES', len(image_ids), image_ids[0], image_ids[-1], flush=True)

    matcher = __import__('models.matcher', fromlist=['HungarianMatcher'])
    hm = matcher.HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

    outpath = pathlib.Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    start = time.time()
    ident = {'n_images': 0, 'logit_maxabs': 0.0, 'box_maxabs': 0.0,
             'mass_sum_min': 9.0, 'mass_sum_max': 0.0}
    nontrivial = []

    with gzip.open(outpath, 'wt', encoding='utf8') as raw, torch.no_grad():
        for ordinal, image_id in enumerate(image_ids, 1):
            coco_info = coco.loadImgs([image_id])[0]
            image = Image.open(str(image_root / coco_info['file_name'])).convert('RGB')
            inputs = processor(images=image, return_tensors='pt')
            pixel_values = inputs['pixel_values'].cuda()
            pixel_mask = inputs['pixel_mask'].cuda()
            hgt, wgt = pixel_values.shape[-2:]
            size = (image.width, image.height)

            attn._d3_bias = None
            attn._d3_capture = {}
            clean = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            weights = attn._d3_capture['weights'][0]
            locations = attn._d3_capture['locations'][0]
            attn._d3_capture = None
            pair_mass = weights.reshape(weights.shape[0], n_heads, n_pairs) / n_heads

            if ordinal <= args.identity_images:
                zero_bias = torch.zeros((1, weights.shape[0], n_heads, n_pairs), device='cuda')
                attn._d3_bias = zero_bias
                zero = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                attn._d3_bias = None
                zero_logits = zero['logits'] if 'logits' in zero else zero['pred_logits']
                clean_logits = clean['logits'] if 'logits' in clean else clean['pred_logits']
                ident['logit_maxabs'] = max(ident['logit_maxabs'],
                                            float((zero_logits - clean_logits).abs().max()))
                ident['box_maxabs'] = max(ident['box_maxabs'],
                                          float((zero['pred_boxes'] - clean['pred_boxes']).abs().max()))
                ident['n_images'] += 1
                sums = pair_mass.reshape(pair_mass.shape[0], -1).sum(-1)
                ident['mass_sum_min'] = min(ident['mass_sum_min'], float(sums.min()))
                ident['mass_sum_max'] = max(ident['mass_sum_max'], float(sums.max()))

            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
                    if not a.get('iscrowd', 0) and a['bbox'][2] > 0 and a['bbox'][3] > 0]
            if not anns:
                counts['images_no_anns'] += 1
                continue
            labels = torch.tensor([a['category_id'] for a in anns], device='cuda', dtype=torch.long)
            gt_rows = []
            for a in anns:
                gt_rows.append([(a['bbox'][0] + a['bbox'][2] / 2) / size[0],
                                (a['bbox'][1] + a['bbox'][3] / 2) / size[1],
                                a['bbox'][2] / size[0], a['bbox'][3] / size[1]])
                a['label_index'] = int(a['category_id'])
            boxes_gt = torch.tensor(gt_rows, device='cuda', dtype=torch.float32).reshape(-1, 4)
            matcher_input = {'pred_logits': clean['logits'] if 'logits' in clean else clean['pred_logits'],
                             'pred_boxes': clean['pred_boxes']}
            idx = hm(matcher_input, [{'labels': labels, 'boxes': boxes_gt}])[0]
            pairs = [(int(q), int(g)) for q, g in zip(idx[0].tolist(), idx[1].tolist())]
            best = {}
            for q, g in pairs:
                ann = anns[g]
                b = area_bin(float(ann['bbox'][2]) * float(ann['bbox'][3]))
                if b not in best or ann['id'] < best[b][2]['id']:
                    best[b] = (q, g, ann)
            for _b in ('small', 'medium', 'large'):
                if _b not in best:
                    counts['no_match_' + _b] += 1
            for bin_name, (query, _g, ann) in best.items():
                counts['candidate_' + bin_name] += 1
                x, y, bw, bh = [float(v) for v in ann['bbox']]
                loc = locations[query].reshape(-1, 2).cpu().numpy()
                loc_px = np.stack([loc[:, 0] * size[0], loc[:, 1] * size[1]], -1)
                inside = ((loc_px[:, 0] >= x) & (loc_px[:, 0] <= x + bw) &
                          (loc_px[:, 1] >= y) & (loc_px[:, 1] <= y + bh))
                in_ids = np.flatnonzero(inside)
                out_ids = np.flatnonzero(~inside)
                mass = pair_mass[query].reshape(-1).cpu().numpy()
                counts['pairs_in_' + bin_name] += int(len(in_ids))
                counts['pairs_total_' + bin_name] += int(len(mass))
                if len(in_ids) < 4 or len(out_ids) < 4:
                    counts['excluded_overlap_' + bin_name] += 1
                    continue
                clean_local = local_quality(clean, query, ann, size)
                inside_mass = float(mass[in_ids].sum())
                base = SELECT_SEED + image_id * 1009 + ann['id'] * 7919
                rng = np.random.Generator(np.random.PCG64(int(base)))
                shuffled_in = rng.permutation(in_ids)
                shuffled_out = rng.permutation(out_ids)
                arms = {'fixed_inside': [(int(j), 1.0) for j in shuffled_in[:4]],
                        'fixed_outside': [(int(j), 1.0) for j in shuffled_out[:4]]}
                for m in MASSES:
                    selected = take_mass_pairs(shuffled_in, mass, m)
                    if selected is None:
                        counts['excluded_mass_%s_%s' % (m, bin_name)] += 1
                        continue
                    arms['mass_%s' % m] = selected

                region_meta = {}
                for side_cells in SIDES:
                    half_c = side_cells * 16 / 2.0
                    hx = half_c * size[0] / wgt
                    hy = half_c * size[1] / hgt
                    centre_in = (x + bw / 2.0, y + bh / 2.0)
                    centre_out = outside_centre(centre_in[0], centre_in[1], bw, bh, hx, hy, size)
                    for tag, centre in (('region%d_inside' % side_cells, centre_in),
                                        ('region%d_outside' % side_cells, centre_out)):
                        if centre is None:
                            counts['region_no_room_%d_%s' % (side_cells, bin_name)] += 1
                            continue
                        rcx, rcy = centre
                        in_region = ((loc_px[:, 0] >= rcx - hx) & (loc_px[:, 0] <= rcx + hx) &
                                     (loc_px[:, 1] >= rcy - hy) & (loc_px[:, 1] <= rcy + hy))
                        ids_r = np.flatnonzero(in_region)
                        zero_heads = 0
                        if len(ids_r):
                            head_counts = np.bincount(ids_r // n_pairs, minlength=n_heads)
                            zero_heads = int((head_counts >= n_pairs).sum())
                        counts['%s_pairs_%s' % (tag, bin_name)] += int(len(ids_r))
                        if len(ids_r) == 0:
                            counts['%s_empty_%s' % (tag, bin_name)] += 1
                        region_meta[tag] = (int(len(ids_r)), zero_heads)
                        arms[tag] = [(int(j), 1.0) for j in ids_r]

                if ordinal <= args.identity_images and len(nontrivial) < 5:
                    bias = torch.zeros((1, weights.shape[0], n_heads, n_pairs), device='cuda')
                    bias[0, query] = float('-inf')
                    attn._d3_bias = bias
                    pert = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                    attn._d3_bias = None
                    nontrivial.append({'image_id': int(image_id), 'annotation_id': int(ann['id']),
                                       'clean': clean_local,
                                       'full_delete': local_quality(pert, query, ann, size)})

                common = {'model': 'deformable_detr_r50', 'image_id': int(image_id),
                          'annotation_id': int(ann['id']), 'query': int(query),
                          'seed': SELECT_SEED, 'area': ann['area'], 'area_bin': bin_name,
                          'layer_index_zero_based': LAYER_INDEX,
                          'box_pairs': len(in_ids), 'outside_pairs': len(out_ids),
                          'clean_in_box_attention_mass': inside_mass,
                          'clean_local': clean_local, 'mode': 'query'}
                for arm_name, cells in arms.items():
                    bias = torch.zeros((1, weights.shape[0], n_heads, n_pairs),
                                       dtype=torch.float32, device='cuda')
                    removed = 0.0
                    for flat, fraction in cells:
                        h_idx = flat // n_pairs
                        p_idx = flat % n_pairs
                        value = float('-inf') if fraction >= 1.0 - 1e-8 else math.log1p(-fraction)
                        bias[0, query, h_idx, p_idx] = value
                        removed += float(mass[flat]) * fraction
                    attn._d3_bias = bias
                    pert = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                    attn._d3_bias = None
                    local = local_quality(pert, query, ann, size)
                    row = dict(common, arm=arm_name, removed_attention_mass=removed,
                               attenuated_pair_count=len(cells),
                               fractional_pair_count=sum(0 < f < 1 for _, f in cells),
                               region_pairs=region_meta.get(arm_name, (0, 0))[0],
                               head_zero_heads=region_meta.get(arm_name, (0, 0))[1],
                               perturbed_local=local)
                    raw.write(json.dumps(row, sort_keys=True, default=float) + '\n')
                    counts['rows_%s_%s' % (arm_name, bin_name)] += 1
            if ordinal % 100 == 0:
                raw.flush()
                print('D3_PROGRESS', ordinal, 'elapsed_s', round(time.time() - start, 1), flush=True)

    sha = hashlib.sha256(outpath.read_bytes()).hexdigest()
    summary = {'model': 'deformable_detr_r50', 'counts': dict(counts),
               'elapsed_s': round(time.time() - start, 1), 'n_images': len(image_ids),
               'identity': ident, 'nontrivial_full_delete': nontrivial,
               'raw_path': str(outpath), 'raw_sha256': sha,
               'attention_geometry': {'layer': LAYER_INDEX, 'heads': n_heads,
                                      'levels': n_levels, 'points': n_points},
               'select_seed': SELECT_SEED, 'masses': list(MASSES), 'sides': list(SIDES)}
    print('D3_DONE', json.dumps(summary, sort_keys=True), flush=True)
    pathlib.Path(str(outpath) + '.summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
