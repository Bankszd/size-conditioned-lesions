"""Task D2: (head, sampling point) damage on DEIMv2-S, query-specific only.

Reuses the frozen Stage-7 statistics (image-level bootstrap, 5000 resamples,
seed 20271001) through stage7_analysis.bootstrap_difference.
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

sys.path.insert(0, '_rel(WORK_ROOT, "")')
sys.path.insert(0, 'experiments/code')

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO

from mve.deimv2_s_loader import load_model

import stage7_analysis as S

HOLDOUT_SHA = 'dab7feec54b51a9f6bad9c1dc25c48f093be83d7b5f7cd15f44adf62fbd036ac'
SELECT_SEED = 20270930
SOURCE = '_rel(REPO_ROOT, "")/vendor/deimv2'
CONFIG = __import__('os').environ.get('D2_CONFIG', '_rel(WORK_ROOT, "")/mve/deimv2_s_hf_config.json')
WEIGHTS = '_rel(CKPT_ROOT, "")/deimv2_s_coco.safetensors'
LAYER_INDEX = 2
MASSES = (0.02, 0.05, 0.10, 0.20, 0.40)
ARMS = ('fixed_inside', 'fixed_outside') + tuple('mass_%s' % m for m in MASSES)


def install_patch():
    from engine.deim.dfine_decoder import MSDeformableAttention

    def patched_forward(self, query, reference_points, value, value_spatial_shapes):
        bs, Len_q = query.shape[:2]
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
        logits = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        bias = getattr(self, '_ffos_bias', None)
        if bias is not None:
            logits = logits + bias
        attention_weights = torch.nan_to_num(F.softmax(logits, dim=-1), nan=0.0)
        if reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        elif reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device, dtype=query.dtype)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                                  + sampling_offsets / offset_normalizer)
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4')
        capture = getattr(self, '_ffos_capture', None)
        if capture is not None:
            capture['weights'] = attention_weights.detach()
            capture['locations'] = sampling_locations.detach()
        return self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations,
                                            attention_weights, self.num_points_list)

    MSDeformableAttention.forward = patched_forward
    return MSDeformableAttention


def area_bin(area):
    return 'small' if area < 32 ** 2 else ('medium' if area < 96 ** 2 else 'large')


def crowd_count(ann, anns):
    x, y, w, h = ann['bbox']
    center = np.array([x + w / 2, y + h / 2])
    radius = 1.5 * float(np.linalg.norm([w, h]))
    return sum(int(np.linalg.norm(np.array([a['bbox'][0] + a['bbox'][2] / 2,
                                            a['bbox'][1] + a['bbox'][3] / 2]) - center) <= radius)
               for a in anns if a['id'] != ann['id'])


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
    logits = outputs['pred_logits'][0]
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
    return {'quality': prob * float(inter / max(union, 1e-12)), 'p_gt': prob,
            'iou': float(inter / max(union, 1e-12))}


def outside_centre(cx, cy, bw, bh, hx, hy, size):
    """Deterministic same-size region fully outside the GT box, kept inside the image."""
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


def boot_two_sample(ga, gb, resamples=5000, seed=20271001):
    """Independent image-level bootstrap of mean(a) - mean(b); non-finite values dropped."""
    av = np.array([np.mean(ga[i]) for i in ga], dtype=float)
    bv = np.array([np.mean(gb[i]) for i in gb], dtype=float)
    dropped = int((~np.isfinite(av)).sum() + (~np.isfinite(bv)).sum())
    av = av[np.isfinite(av)]
    bv = bv[np.isfinite(bv)]
    if len(av) < 2 or len(bv) < 2:
        return {'status': 'insufficient_data', 'n_a': int(len(av)), 'n_b': int(len(bv)),
                'n_dropped_nonfinite': dropped}
    rng = np.random.Generator(np.random.PCG64(seed))
    da = av[rng.integers(0, len(av), size=(resamples, len(av)))].mean(axis=1)
    db = bv[rng.integers(0, len(bv), size=(resamples, len(bv)))].mean(axis=1)
    boot = da - db
    return {'estimate': float(av.mean() - bv.mean()),
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'n_a': int(len(av)), 'n_b': int(len(bv)), 'n_dropped_nonfinite': dropped,
            'bootstrap_resamples': resamples, 'bootstrap_seed': seed,
            'mean_a': float(av.mean()), 'mean_b': float(bv.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--tag', default='full')
    parser.add_argument('--out-dir', default='_rel(OUT_ROOT, "")')
    parser.add_argument('--data', choices=['coco', 'voc'], default='coco')
    parser.add_argument('--voc-json', default='_rel(DATA_ROOT, "")/voc2007test/instances_voc2007test.json')
    parser.add_argument('--voc-images', default='_rel(DATA_ROOT, "")/VOCdevkit/VOC2007/JPEGImages')
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    global CONFIG
    if args.config:
        CONFIG = args.config
    print('N2_CONFIG_USED', json.dumps({'path': CONFIG,
          'sha256': hashlib.sha256(open(CONFIG, 'rb').read()).hexdigest()}), flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(SELECT_SEED)
    print('D2_CONFIG', json.dumps(vars(args), sort_keys=True), flush=True)
    model = load_model(SOURCE, CONFIG, WEIGHTS, device='cuda').eval()
    weights_sha = hashlib.sha256(open(WEIGHTS, 'rb').read()).hexdigest()
    print('D2_WEIGHTS_SHA256', weights_sha, flush=True)
    from engine.data.dataset import mscoco_category2label
    from engine.deim.matcher import HungarianMatcher
    matcher = HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                               use_focal_loss=True, alpha=0.25, gamma=2.0)
    cross_attn = model.decoder.decoder.layers[LAYER_INDEX].cross_attn
    n_heads = cross_attn.num_heads
    n_pairs = sum(cross_attn.num_points_list)
    print('D2_ATTENTION', json.dumps({'n_heads': n_heads, 'n_pairs': n_pairs,
                                      'num_points_list': list(cross_attn.num_points_list)}), flush=True)

    # Reference output before patching, for the zero-dose identity check.
    if args.data == 'voc':
        coco = COCO(args.voc_json)
        image_ids = sorted(int(x) for x in coco.getImgIds())
        image_root = args.voc_images
    else:
        coco = COCO('_rel(DATA_ROOT, "")/coco/annotations/instances_val2017.json')
        permutation = np.random.Generator(np.random.PCG64(20270925)).permutation(sorted(coco.getImgIds()))
        image_ids = sorted(int(x) for x in permutation[1000:5000])
        image_root = '_rel(DATA_ROOT, "")/coco/val2017'
    digest = hashlib.sha256(','.join(str(x) for x in image_ids).encode()).hexdigest()
    if args.data != 'voc':
        assert digest == HOLDOUT_SHA
    if args.limit:
        image_ids = image_ids[:args.limit]
    print('D2_HOLDOUT_SHA256', digest, 'N', len(image_ids), flush=True)

    transform = T.Compose([T.Resize((640, 640)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    info0 = coco.loadImgs([image_ids[0]])[0]
    image0 = Image.open(image_root + '/' + info0['file_name']).convert('RGB')
    tensor0 = transform(image0).unsqueeze(0).cuda()
    with torch.no_grad():
        reference = model.raw_forward(tensor0)
    install_patch()
    cross_attn._ffos_bias = None
    cross_attn._ffos_capture = None
    with torch.no_grad():
        patched = model.raw_forward(tensor0)
    diff_logits = float((patched['pred_logits'] - reference['pred_logits']).abs().max().item())
    diff_boxes = float((patched['pred_boxes'] - reference['pred_boxes']).abs().max().item())
    zero_bias = torch.zeros((1, 300, n_heads, n_pairs), device='cuda')
    cross_attn._ffos_bias = zero_bias
    with torch.no_grad():
        zero = model.raw_forward(tensor0)
    cross_attn._ffos_bias = None
    zero_logits = float((zero['pred_logits'] - reference['pred_logits']).abs().max().item())
    zero_boxes = float((zero['pred_boxes'] - reference['pred_boxes']).abs().max().item())
    capture = {}
    cross_attn._ffos_capture = capture
    with torch.no_grad():
        _ = model.raw_forward(tensor0)
    cross_attn._ffos_capture = None
    weights = capture['weights'][0]
    locations = capture['locations'][0]
    full_bias = torch.zeros((1, 300, n_heads, n_pairs), dtype=torch.float32, device='cuda')
    for head_index in range(n_heads):
        full_bias[0, 0, head_index, 0:4] = float('-inf')
    cross_attn._ffos_bias = full_bias
    with torch.no_grad():
        nontrivial = model.raw_forward(tensor0)
    cross_attn._ffos_bias = None
    nontrivial_logits = float((nontrivial['pred_logits'] - reference['pred_logits']).abs().max().item())
    print('D2_SANITY', json.dumps({'patch_no_bias_logits': diff_logits, 'patch_no_bias_boxes': diff_boxes,
                                   'zero_bias_logits': zero_logits, 'zero_bias_boxes': zero_boxes,
                                   'full_delete_logits': nontrivial_logits}, sort_keys=True), flush=True)
    assert diff_logits == 0.0 and diff_boxes == 0.0
    assert zero_logits == 0.0 and zero_boxes == 0.0
    assert nontrivial_logits > 0.0

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / ('task_d2_%s.jsonl.gz' % args.tag)
    counts = collections.Counter()
    rows = []
    start = time.time()
    with gzip.open(raw_path, 'wt', encoding='utf8', compresslevel=6) as raw, torch.no_grad():
        for ordinal, image_id in enumerate(image_ids, 1):
            info = coco.loadImgs([image_id])[0]
            image = Image.open(image_root + '/' + info['file_name']).convert('RGB')
            tensor = transform(image).unsqueeze(0).cuda()
            size = (image.width, image.height)
            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
                    if not a.get('iscrowd', 0) and a['bbox'][2] > 0 and a['bbox'][3] > 0]
            capture = {}
            cross_attn._ffos_capture = capture
            outputs = model.raw_forward(tensor)
            cross_attn._ffos_capture = None
            weights = capture['weights'][0]
            locations = capture['locations'][0].reshape(weights.shape[0], n_heads, n_pairs, 2)
            boxes = torch.tensor([[(a['bbox'][0] + a['bbox'][2] / 2) / size[0],
                                   (a['bbox'][1] + a['bbox'][3] / 2) / size[1],
                                   a['bbox'][2] / size[0], a['bbox'][3] / size[1]] for a in anns],
                                 device='cuda', dtype=torch.float32).reshape(-1, 4)
            labels = torch.tensor([mscoco_category2label[a['category_id']] for a in anns],
                                  device='cuda', dtype=torch.long)
            if len(anns) == 0:
                counts['no_annotations'] += 1
                continue
            indices = matcher(outputs, [{'labels': labels, 'boxes': boxes}])['indices'][0]
            chosen = {}
            for q, gi in zip(indices[0].tolist(), indices[1].tolist()):
                ann = dict(anns[gi])
                ann['label_index'] = int(labels[gi].item())
                b = area_bin(ann['area'])
                if b not in chosen or ann['id'] < chosen[b]['id']:
                    chosen[b] = ann
                    ann['query'] = int(q)
            for bin_name in ('small', 'medium', 'large'):
                if bin_name not in chosen:
                    continue
                ann = chosen[bin_name]
                query = ann['query']
                x, y, w, h = ann['bbox']
                loc = locations[query].reshape(n_heads, n_pairs, 2).clone()
                loc[..., 0] *= size[0]
                loc[..., 1] *= size[1]
                inside = (loc[..., 0] >= x) & (loc[..., 0] <= x + w) & \
                         (loc[..., 1] >= y) & (loc[..., 1] <= y + h)
                inside = inside.reshape(-1).cpu()
                in_ids = np.flatnonzero(inside)
                out_ids = np.flatnonzero(~inside)
                pair_mass = (weights[query].reshape(-1) / n_heads).cpu().numpy()
                counts['candidate_' + bin_name] += 1
                if len(in_ids) < 4 or len(out_ids) < 4:
                    counts['excluded_overlap_' + bin_name] += 1
                    continue
                clean_local = local_quality(outputs, query, ann, size)
                inside_mass = float(pair_mass[in_ids].sum())
                # Task N: fixed absolute square regions whose side is 2 or 3 stride-16 cells in
                # the model input frame (640 px), centred on the GT box centre / on a point
                # outside the GT box for the control arm.
                loc_flat = loc.reshape(-1, 2)
                region_arms = {}
                region_meta = {}
                for side_cells in (2, 3):
                    half_c = side_cells * 16 / 2.0
                    hx = half_c * size[0] / 640.0
                    hy = half_c * size[1] / 640.0
                    centre_in = (x + w / 2.0, y + h / 2.0)
                    centre_out = outside_centre(centre_in[0], centre_in[1], w, h, hx, hy, size)
                    for tag_r, centre in (('region%d_inside' % side_cells, centre_in),
                                          ('region%d_outside' % side_cells, centre_out)):
                        if centre is None:
                            counts['region_no_room_%d_%s' % (side_cells, bin_name)] += 1
                            continue
                        rcx, rcy = centre
                        in_region = ((loc_flat[:, 0] >= rcx - hx) & (loc_flat[:, 0] <= rcx + hx) &
                                     (loc_flat[:, 1] >= rcy - hy) & (loc_flat[:, 1] <= rcy + hy))
                        ids_r = np.flatnonzero(in_region.cpu().numpy())
                        counts['%s_pairs_%s' % (tag_r, bin_name)] += int(len(ids_r))
                        if len(ids_r) == 0:
                            counts['%s_empty_%s' % (tag_r, bin_name)] += 1
                        zero_heads = 0
                        if len(ids_r):
                            head_counts = np.bincount(ids_r // n_pairs, minlength=n_heads)
                            if head_counts.max() >= n_pairs:
                                counts['%s_headfull_%s' % (tag_r, bin_name)] += 1
                            zero_heads = int((head_counts >= n_pairs).sum())
                        region_arms[tag_r] = [(int(j), 1.0) for j in ids_r]
                        region_meta[tag_r] = (int(len(ids_r)), zero_heads)
                base = SELECT_SEED + image_id * 1009 + ann['id'] * 7919
                rng = np.random.Generator(np.random.PCG64(int(base)))
                shuffled_in = rng.permutation(in_ids)
                shuffled_out = rng.permutation(out_ids)
                arms = {'fixed_inside': [(int(j), 1.0) for j in shuffled_in[:4]],
                        'fixed_outside': [(int(j), 1.0) for j in shuffled_out[:4]]}
                for mass in MASSES:
                    selected = take_mass_pairs(shuffled_in, pair_mass, mass)
                    if selected is None:
                        counts['excluded_mass_%s_%s' % (mass, bin_name)] += 1
                        continue
                    arms['mass_%s' % mass] = selected
                arms.update(region_arms)
                common = {'model': 'deimv2_s_coco', 'image_id': image_id, 'annotation_id': ann['id'],
                          'query': query, 'seed': SELECT_SEED, 'area': ann['area'],
                          'area_bin': bin_name, 'crowd_count': crowd_count(ann, anns),
                          'box_cells': len(in_ids), 'outside_pairs': len(out_ids),
                          'clean_in_box_attention_mass': inside_mass,
                          'clean_local': clean_local, 'mode': 'query'}
                for arm_name, pairs in arms.items():
                    bias = torch.full((1, 300, n_heads, n_pairs), 0.0, dtype=torch.float32, device='cuda')
                    removed = 0.0
                    for flat, fraction in pairs:
                        h_idx = flat // n_pairs
                        p_idx = flat % n_pairs
                        value = float('-inf') if fraction >= 1.0 - 1e-8 else math.log1p(-fraction)
                        bias[0, query, h_idx, p_idx] = value
                        removed += pair_mass[flat] * fraction
                    assert len(pairs) <= n_heads * n_pairs
                    cross_attn._ffos_bias = bias
                    perturbed = model.raw_forward(tensor)
                    cross_attn._ffos_bias = None
                    local = local_quality(perturbed, query, ann, size)
                    row = dict(common, arm=arm_name, removed_attention_mass=removed,
                               attenuated_cell_count=len(pairs),
                               region_pairs=region_meta.get(arm_name, (0, 0))[0],
                               head_zero_heads=region_meta.get(arm_name, (0, 0))[1],
                               fractional_cell_count=sum(0 < f < 1 for _, f in pairs),
                               perturbed_local=local)
                    raw.write(json.dumps(row, sort_keys=True, default=float) + '\n')
                    rows.append(row)
                    counts['rows_%s_%s' % (arm_name, bin_name)] += 1
            if ordinal % 50 == 0:
                raw.flush()
                print('D2_PROGRESS', ordinal, 'elapsed_s', round(time.time() - start, 1), flush=True)
    print('D2_INFERENCE_DONE', json.dumps({'elapsed_s': round(time.time() - start, 1),
                                           'raw_path': str(raw_path),
                                           's_per_image': round((time.time() - start) / len(image_ids), 3),
                                           'counts': dict(counts)}, sort_keys=True), flush=True)

    # Frozen Stage-7 statistics on the paired local-quality readout.
    paired = collections.defaultdict(dict)
    for row in rows:
        paired[(row['image_id'], row['annotation_id'])][row['arm']] = row
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    descriptive = collections.defaultdict(list)
    for (image_id, _ann), arms in paired.items():
        if 'fixed_inside' not in arms or 'fixed_outside' not in arms:
            continue
        inside = arms['fixed_inside']
        outside = arms['fixed_outside']
        d4 = outside['perturbed_local']['quality'] - inside['perturbed_local']['quality']
        groups[('fixed_d4', inside['area_bin'])][image_id].append(d4)
        if 'mass_0.02' in arms:
            l02 = inside['clean_local']['quality'] - arms['mass_0.02']['perturbed_local']['quality']
            groups[('mass_0.02', inside['area_bin'])][image_id].append(l02)
        for side_cells in (2, 3):
            inside_name = 'region%d_inside' % side_cells
            outside_name = 'region%d_outside' % side_cells
            if inside_name in arms:
                rw = arms[inside_name]
                loss = rw['clean_local']['quality'] - rw['perturbed_local']['quality']
                groups[('region%d_loss' % side_cells, inside['area_bin'])][image_id].append(loss)
                descriptive[(inside_name, inside['area_bin'], 'removed_mass')].append(rw['removed_attention_mass'])
                descriptive[(inside_name, inside['area_bin'], 'quality')].append(rw['perturbed_local']['quality'])
                descriptive[(inside_name, inside['area_bin'], 'n_pairs')].append(rw['attenuated_cell_count'])
            if inside_name in arms and outside_name in arms:
                dd = arms[outside_name]['perturbed_local']['quality'] - arms[inside_name]['perturbed_local']['quality']
                groups[('region%d_d' % side_cells, inside['area_bin'])][image_id].append(dd)
                descriptive[(outside_name, inside['area_bin'], 'removed_mass')].append(
                    arms[outside_name]['removed_attention_mass'])
                descriptive[(outside_name, inside['area_bin'], 'quality')].append(
                    arms[outside_name]['perturbed_local']['quality'])
        for arm in ('fixed_inside', 'fixed_outside', 'mass_0.02', 'mass_0.4',
                    'region2_inside', 'region2_outside', 'region3_inside', 'region3_outside'):
            if arm in arms:
                descriptive[(arm, inside['area_bin'], 'removed_mass')].append(arms[arm]['removed_attention_mass'])
                descriptive[(arm, inside['area_bin'], 'quality')].append(arms[arm]['perturbed_local']['quality'])
    tests = {}
    for name in ('fixed_d4', 'mass_0.02'):
        gs = groups[(name, 'small')]
        gl = groups[(name, 'large')]
        counts['test_%s_n_small' % name] = len(gs)
        counts['test_%s_n_large' % name] = len(gl)
        if len(gs) < 25 or len(gl) < 25:
            tests[name + '_small_minus_large'] = {'status': 'insufficient_data',
                                                  'n_small': len(gs), 'n_large': len(gl)}
            continue
        tests[name + '_small_minus_large'] = S.bootstrap_difference(image_ids, gs, gl)
    for side_cells in (2, 3):
        tests['region%d_loss_small_minus_large' % side_cells] = boot_two_sample(
            groups[('region%d_loss' % side_cells, 'small')],
            groups[('region%d_loss' % side_cells, 'large')])
        tests['region%d_d_small_minus_large' % side_cells] = boot_two_sample(
            groups[('region%d_d' % side_cells, 'small')],
            groups[('region%d_d' % side_cells, 'large')])
    report = {'status': 'post_review_exploratory', 'model': 'deimv2_s_coco',
              'weights_sha256': weights_sha, 'holdout_sha256': digest,
              'n_images': len(image_ids), 'layer_index_zero_based': LAYER_INDEX,
              'n_heads': n_heads, 'n_pairs_per_head': n_pairs,
              'counts': dict(counts), 'tests': tests,
              'descriptive': {'/'.join(k): {'mean': float(np.mean(v)), 'n': len(v)}
                              for k, v in descriptive.items()}}
    out_path = out_dir / ('task_d2_%s_summary.json' % args.tag)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + '\n')
    print('D2_TESTS', json.dumps(tests, sort_keys=True, default=float), flush=True)
    print('D2_SUMMARY', str(out_path), hashlib.sha256(out_path.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()