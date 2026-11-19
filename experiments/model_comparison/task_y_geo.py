"""Task C v2: detector-level, size-decomposed AP readout under query-specific damage.

Vectorised mask construction: all masks for one image are built with numpy and
transferred to the device once per forward, instead of per-cell tensor writes.
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
import pathlib
import sys
import time

sys.path.insert(0, '.')
import compat_shim  # noqa: F401
sys.path.insert(0, 'experiments/code')

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from intervene import load_model, per_query_result
from mve import file_sha256

HOLDOUT_SHA = 'dab7feec54b51a9f6bad9c1dc25c48f093be83d7b5f7cd15f44adf62fbd036ac'
SELECT_SEED = 20270930
BOOT_SEED = 20271004
BOOT_RESAMPLES = 500
MASSES = (0.02, 0.05, 0.10, 0.20, 0.40)
PROPS = tuple(float(x) for x in os.environ.get('Y_PROPS', '0.50').split(','))
INSIDE_ARMS = ('in4', 'in_m0.02', 'in_geo0.50') + tuple('in_prop%0.2f' % p for p in PROPS)
OUTSIDE_ARMS = ()
ARMS = ('clean',) + INSIDE_ARMS + OUTSIDE_ARMS
VERSIONS = ('layer2', 'all_layers')


class LayerMaskController:
    def __init__(self, modules):
        self.modules = modules
        self.masks = [None] * len(modules)
        self.capture = None
        self.pre_handles = []
        self.post_handles = []
        for index, module in enumerate(modules):
            self.pre_handles.append(module.register_forward_pre_hook(self._before(index), with_kwargs=True))
            self.post_handles.append(module.register_forward_hook(self._after(index)))

    def _before(self, index):
        def hook(_module, args, kwargs):
            arrays = self.masks[index]
            if arrays is None:
                return args, kwargs
            query = kwargs.get('query', args[0] if len(args) > 0 else None)
            key = kwargs.get('key', args[1] if len(args) > 1 else None)
            assert query is not None and key is not None
            nq, ns = query.shape[0], key.shape[0]
            rows, cols, values = arrays
            bias_np = np.zeros((nq, ns), dtype=np.float32)
            if len(rows):
                bias_np[rows, cols] = values
            bias = torch.from_numpy(bias_np).to(device=query.device, dtype=query.dtype)
            kwargs['attn_mask'] = bias
            return args, kwargs
        return hook

    def _after(self, index):
        def hook(_module, _args, output):
            if self.capture is not None:
                self.capture[index] = output[1].detach().float().cpu().numpy()[0]
        return hook

    def clean(self, model, tensor):
        self.masks = [None] * len(self.modules)
        self.capture = [None] * len(self.modules)
        try:
            output = model([tensor])
            return output, list(self.capture)
        finally:
            self.capture = None

    def run(self, model, tensor, masks):
        self.masks = masks
        try:
            return model([tensor])
        finally:
            self.masks = [None] * len(self.modules)

    def close(self):
        for handle in self.pre_handles + self.post_handles:
            handle.remove()


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
        records = [record for record in records if record is not None]
        if not records:
            continue
        gt_ignore = np.concatenate([record['gtIgnore'] for record in records])
        positive_gt = np.count_nonzero(gt_ignore == 0)
        if positive_gt == 0:
            continue
        scores = np.concatenate([record['dtScores'][:max_det] for record in records])
        sorting = np.argsort(-scores, kind='mergesort')
        matches = np.concatenate([record['dtMatches'][:, :max_det] for record in records], axis=1)[:, sorting]
        ignored = np.concatenate([record['dtIgnore'][:, :max_det] for record in records], axis=1)[:, sorting]
        true_positive = np.logical_and(matches, np.logical_not(ignored))
        false_positive = np.logical_and(np.logical_not(matches), np.logical_not(ignored))
        true_cumsum = np.cumsum(true_positive, axis=1).astype(float)
        false_cumsum = np.cumsum(false_positive, axis=1).astype(float)
        precision = np.zeros((iou_count, recall_count), dtype=float)
        for iou_index, (true_row, false_row) in enumerate(zip(true_cumsum, false_cumsum)):
            recall = true_row / positive_gt
            values = true_row / (true_row + false_row + np.spacing(1))
            if len(values):
                values = np.maximum.accumulate(values[::-1])[::-1]
                positions = np.searchsorted(recall, params.recThrs, side='left')
                valid = positions < len(values)
                precision[iou_index, valid] = values[positions[valid]]
        category_values.append(precision.mean())
    return float(np.mean(category_values))


def mass_selection(order, weights, target):
    w = weights[order]
    cumulative = np.cumsum(w)
    if cumulative[-1] < target - 1e-7:
        return None
    index = int(np.searchsorted(cumulative, target - 1e-10, side='left'))
    index = min(index, len(order) - 1)
    previous = cumulative[index - 1] if index > 0 else 0.0
    fraction = (target - previous) / max(float(w[index]), 1e-12)
    fraction = float(min(max(fraction, 0.0), 1.0))
    cells = [(int(order[i]), 1.0) for i in range(index)]
    if fraction > 0:
        cells.append((int(order[index]), fraction))
    return cells


def cells_to_arrays(cells):
    if not cells:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    cols = np.asarray([j for j, _ in cells], dtype=np.int64)
    values = np.asarray([f for _, f in cells], dtype=np.float32)
    values = np.where(values >= 1.0 - 1e-8, -np.inf, np.log1p(-values)).astype(np.float32)
    return cols, values


def build_arm_mask(per_query_cells, nq):
    cols_all, vals_all, rows_all = [], [], []
    for qi, cells in per_query_cells.items():
        cols, values = cells_to_arrays(cells)
        if len(cols) == 0:
            continue
        rows_all.append(np.full(len(cols), qi, dtype=np.int64))
        cols_all.append(cols)
        vals_all.append(values)
    if not cols_all:
        return None
    return (np.concatenate(rows_all), np.concatenate(cols_all), np.concatenate(vals_all))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--model', choices=['detr', 'conditional_detr'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--skip-bootstrap', action='store_true')
    parser.add_argument('--out-dir', default='_rel(OUT_ROOT, "")')
    parser.add_argument('--tag-suffix', default='')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SELECT_SEED)
    print('TASK_C_CONFIG', json.dumps(vars(args), sort_keys=True), flush=True)
    model, post, _module, transform = load_model(args.model, args.cache)
    ck_sha = file_sha256(pathlib.Path(args.cache) / (args.model + '.pth'))
    print('TASK_C_CHECKPOINT_SHA256', args.model, ck_sha, flush=True)

    decoder = model.transformer.decoder
    attention_modules = [layer.multihead_attn if args.model == 'detr' else layer.cross_attn
                         for layer in decoder.layers]
    print('TASK_C_DECODER_LAYERS', len(attention_modules), flush=True)
    controller = LayerMaskController(attention_modules)
    holder = {}
    def feature_shape(_m, _i, output):
        holder['shape'] = tuple(output[0][-1].tensors.shape[-2:])
    shape_hook = model.backbone.register_forward_hook(feature_shape)

    coco = COCO(str(pathlib.Path(args.cache) / 'coco/annotations/instances_val2017.json'))
    permutation = np.random.Generator(np.random.PCG64(20270925)).permutation(sorted(coco.getImgIds()))
    image_ids = sorted(int(x) for x in permutation[1000:5000])
    digest = hashlib.sha256(','.join(str(x) for x in image_ids).encode()).hexdigest()
    assert digest == HOLDOUT_SHA, digest
    print('TASK_C_HOLDOUT_IDS_SHA256', digest, 'N', len(image_ids), flush=True)
    if args.limit:
        image_ids = image_ids[:args.limit]
    if args.end:
        image_ids = image_ids[args.start:args.end]

    dets = {(version, arm): [] for version in VERSIONS for arm in ARMS}
    counts = collections.Counter()
    dose_acc = collections.defaultdict(float)
    dose_err = collections.defaultdict(float)
    dose_cnt = collections.Counter()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model + ('_smoke%d' % args.limit if args.limit else '_full') + args.tag_suffix
    raw_path = out_dir / ('task_c_%s.jsonl.gz' % tag)
    diag_path = out_dir / ('task_c_%s_diag.jsonl.gz' % tag)
    start = time.time()
    last_report = start
    with gzip.open(raw_path, 'wt', encoding='utf8', compresslevel=6) as raw, \
         gzip.open(diag_path, 'wt', encoding='utf8', compresslevel=6) as diag_file, torch.no_grad():
        for ordinal, image_id in enumerate(image_ids, 1):
            info = coco.loadImgs([image_id])[0]
            image = Image.open(str(pathlib.Path(args.cache) / 'coco/val2017' / info['file_name'])).convert('RGB')
            tensor, _ = transform(image, None)
            tensor = tensor.cuda()
            clean, clean_weights = controller.clean(model, tensor)
            shape = holder['shape']
            size = (image.height, image.width)
            fh, fw = shape
            ih, iw = size
            yy, xx = np.indices((fh, fw))
            xs = ((xx.ravel() + .5) / fw * iw)
            ys = ((yy.ravel() + .5) / fh * ih)
            ns = fh * fw
            query_result = per_query_result(clean, args.model, size)
            boxes = query_result['boxes'].detach().cpu().numpy()
            nq = int(boxes.shape[0])
            counts['images'] += 1
            row = {'model': args.model, 'image_id': image_id, 'seed': SELECT_SEED,
                   'shape': list(shape), 'size': list(size),
                   'n_queries': nq, 'arms': {}}

            clean_result = post(clean, torch.tensor([size], device='cuda'))[0]
            clean_dets = [{'image_id': image_id, 'category_id': int(label),
                           'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                           'score': float(score)}
                          for score, label, b in zip(clean_result['scores'].cpu().tolist(),
                                                     clean_result['labels'].cpu().tolist(),
                                                     clean_result['boxes'].cpu().tolist())]
            for version in VERSIONS:
                dets[(version, 'clean')].extend(clean_dets)
                row['arms'].setdefault(version, {})['clean'] = clean_dets

            if ordinal == 1:
                zero_masks = [(np.array([0]), np.array([0]), np.zeros(1, dtype=np.float32))
                              for _ in attention_modules]
                zero = controller.run(model, tensor, zero_masks)
                zero_logits = float((zero['pred_logits'] - clean['pred_logits']).abs().max().item())
                zero_boxes = float((zero['pred_boxes'] - clean['pred_boxes']).abs().max().item())
                box0 = boxes[0]
                inside0 = np.flatnonzero((xs >= box0[0]) & (xs <= box0[2]) & (ys >= box0[1]) & (ys <= box0[3]))[:4]
                probe_cells = [(int(j), 1.0) for j in inside0]
                probe_cols, probe_vals = cells_to_arrays(probe_cells)
                probe_mask = (np.zeros(len(probe_cols), dtype=np.int64), probe_cols, probe_vals)
                probe = controller.run(model, tensor, [probe_mask for _ in attention_modules])
                probe_logits = float((probe['pred_logits'] - clean['pred_logits']).abs().max().item())
                print('TASK_C_SANITY', json.dumps({'zero_dose_logits': zero_logits, 'zero_dose_boxes': zero_boxes,
                                                   'full_delete_logits': probe_logits}, sort_keys=True), flush=True)
                assert zero_logits == 0.0 and zero_boxes == 0.0
                assert probe_logits > 0.0

            # Per-query geometry, computed once.
            query_cells = {}
            for qi in range(nq):
                box = boxes[qi]
                inside = np.flatnonzero((xs >= box[0]) & (xs <= box[2]) & (ys >= box[1]) & (ys <= box[3]))
                outside = np.flatnonzero(~((xs >= box[0]) & (xs <= box[2]) & (ys >= box[1]) & (ys <= box[3])))
                if len(inside) == 0 or len(outside) == 0 or len(inside) >= ns:
                    counts['skipped_empty_region'] += 1
                    continue
                rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SELECT_SEED, int(image_id), int(qi)])))
                order_in = rng.permutation(inside)
                order_out = rng.permutation(outside)
                area_px = float((box[2] - box[0]) * (box[3] - box[1]))
                dose_bin = 'small' if area_px < 32 ** 2 else ('medium' if area_px < 96 ** 2 else 'large')
                query_cells[qi] = {'inside': inside, 'outside': outside, 'order_in': order_in,
                                   'order_out': order_out, 'dose_bin': dose_bin}
                counts['queries_total'] += 1
                if len(inside) < 4:
                    counts['in4_insufficient'] += 1
                if len(outside) < 4:
                    counts['out4_insufficient'] += 1

            for version in VERSIONS:
                layers = [2] if version == 'layer2' else list(range(len(attention_modules)))
                arm_masks = {}
                arm_cells = {}
                for arm in INSIDE_ARMS + OUTSIDE_ARMS:
                    if arm == 'in_geo0.50':
                        per_layer = [None] * len(attention_modules)
                        per_layer_cells = {}
                        for layer in layers:
                            per_query = {}
                            for qi, geometry in query_cells.items():
                                inside = geometry['inside']
                                n = len(inside)
                                if n == 0:
                                    per_query[qi] = []
                                    if layer == layers[0]:
                                        counts['geo_n0'] += 1
                                    continue
                                whole = int(math.floor(0.5 * n))
                                cells = [(int(geometry['order_in'][i]), 1.0) for i in range(whole)]
                                if n % 2 == 1:
                                    cells.append((int(geometry['order_in'][whole]), 0.5))
                                per_query[qi] = cells
                            per_layer[layer] = build_arm_mask(per_query, nq)
                            per_layer_cells[layer] = per_query
                        arm_masks[arm] = per_layer
                        arm_cells[arm] = per_layer_cells
                    elif arm in ('in4', 'out4', 'in4_strict', 'out4_strict', 'in_frac25'):
                        per_query = {}
                        for qi, geometry in query_cells.items():
                            inside = geometry['inside']
                            outside = geometry['outside']
                            if arm == 'in4':
                                per_query[qi] = ([(int(j), 1.0) for j in inside] if len(inside) < 4
                                                 else [(int(j), 1.0) for j in geometry['order_in'][:4]])
                            elif arm == 'out4':
                                per_query[qi] = ([(int(j), 1.0) for j in outside] if len(outside) < 4
                                                 else [(int(j), 1.0) for j in geometry['order_out'][:4]])
                            elif arm == 'in4_strict':
                                if len(inside) < 4 or len(outside) < 4:
                                    continue
                                per_query[qi] = [(int(j), 1.0) for j in geometry['order_in'][:4]]
                            elif arm == 'out4_strict':
                                if len(inside) < 4 or len(outside) < 4:
                                    continue
                                per_query[qi] = [(int(j), 1.0) for j in geometry['order_out'][:4]]
                            else:
                                count = 0.25 * len(inside)
                                whole = int(math.floor(count))
                                fraction = count - whole
                                cells = [(int(j), 1.0) for j in geometry['order_in'][:whole]]
                                if fraction > 1e-9:
                                    cells.append((int(geometry['order_in'][whole]), float(fraction)))
                                per_query[qi] = cells
                        arrays = build_arm_mask(per_query, nq)
                        arm_masks[arm] = [arrays if index in layers else None
                                          for index in range(len(attention_modules))]
                        arm_cells[arm] = {layer: per_query for layer in layers}
                    else:
                        prop = float(arm.split('in_prop')[1]) if arm.startswith('in_prop') else None
                        mass = None if prop is not None else float(arm.split('_m')[1])
                        inside_arm = arm.startswith('in_m')
                        per_layer = [None] * len(attention_modules)
                        per_layer_cells = {}
                        for layer in layers:
                            weights_all = clean_weights[layer]
                            per_query = {}
                            for qi, geometry in query_cells.items():
                                weights = weights_all[qi]
                                if prop is not None:
                                    target = float(prop) * float(np.sum(weights[geometry['inside']]))
                                    selected = mass_selection(geometry['order_in'], weights, target)
                                    if selected is None:
                                        selected = [(int(j), 1.0) for j in geometry['inside']]
                                elif inside_arm:
                                    selected = mass_selection(geometry['order_in'], weights, mass)
                                    if selected is None:
                                        selected = [(int(j), 1.0) for j in geometry['inside']]
                                        if layer == layers[0]:
                                            counts[arm + '_fallback'] += 1
                                else:
                                    selected = mass_selection(geometry['order_out'], weights, mass)
                                    if selected is None:
                                        selected = [(int(j), 1.0) for j in geometry['outside']]
                                        if layer == layers[0]:
                                            counts[arm + '_fallback'] += 1
                                per_query[qi] = selected
                            per_layer[layer] = build_arm_mask(per_query, nq)
                            per_layer_cells[layer] = per_query
                        arm_masks[arm] = per_layer
                        arm_cells[arm] = per_layer_cells
                if version == 'layer2':
                    weights2 = clean_weights[2]
                    diag = []
                    for qi, geometry in query_cells.items():
                        inside = geometry['inside']
                        n = len(inside)
                        in_mass = float(np.sum(weights2[qi][inside])) if n else 0.0
                        geo_cells = arm_cells['in_geo0.50'][2].get(qi, [])
                        prop_cells = arm_cells.get('in_prop0.50', {}).get(2, {}).get(qi, [])
                        geo_mass = float(sum(float(weights2[qi][j]) * f for j, f in geo_cells))
                        diag.append([n,
                                     (geo_mass / in_mass) if in_mass > 0 else 0.0,
                                     (sum(f for _, f in geo_cells) / n) if n else 0.0,
                                     (sum(f for _, f in prop_cells) / n) if n else 0.0])
                    row['query_diag'] = diag
                for arm in INSIDE_ARMS + OUTSIDE_ARMS:
                    for layer in layers:
                        cells_by_query = arm_cells[arm].get(layer)
                        if not cells_by_query:
                            continue
                        weights_all = clean_weights[layer]
                        for qi, cells in cells_by_query.items():
                            mass = float(sum(weights_all[qi][j] * f for j, f in cells))
                            key = (arm, query_cells[qi]['dose_bin'], layer)
                            dose_acc[key] += mass
                            dose_cnt[key] += 1
                            if arm.startswith('in_prop'):
                                _tgt = float(arm.split('in_prop')[1]) * float(np.sum(weights_all[qi][query_cells[qi]['inside']]))
                                dose_err[arm] = max(dose_err[arm], abs(mass - _tgt))
                for arm in INSIDE_ARMS + OUTSIDE_ARMS:
                    output = controller.run(model, tensor, arm_masks[arm])
                    result = post(output, torch.tensor([size], device='cuda'))[0]
                    predictions = [{'image_id': image_id, 'category_id': int(label),
                                    'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                                    'score': float(score)}
                                   for score, label, b in zip(result['scores'].cpu().tolist(),
                                                              result['labels'].cpu().tolist(),
                                                              result['boxes'].cpu().tolist())]
                    dets[(version, arm)].extend(predictions)
                    row['arms'][version][arm] = predictions
            raw.write(json.dumps(row, sort_keys=True) + '\n')
            diag_file.write(json.dumps({'image_id': image_id, 'model': args.model,
                                        'query_diag': row.get('query_diag', [])}, sort_keys=True) + '\n')
            if ordinal % 20 == 0:
                raw.flush()
                now = time.time()
                print('TASK_C_PROGRESS', args.model, ordinal,
                      'elapsed_s', round(now - start, 1),
                      's_per_image', round((now - start) / ordinal, 3), flush=True)

    shape_hook.remove()
    controller.close()
    print('TASK_C_INFERENCE_DONE', json.dumps({'elapsed_s': round(time.time() - start, 1),
                                               'raw_path': str(raw_path),
                                               's_per_image': round((time.time() - start) / len(image_ids), 3)}),
          flush=True)

    results = {'status': 'post_review_exploratory', 'model': args.model,
               'checkpoint_sha256': ck_sha, 'holdout_sha256': digest,
               'n_images': len(image_ids), 'select_seed': SELECT_SEED,
               'bootstrap_seed': BOOT_SEED, 'bootstrap_resamples': BOOT_RESAMPLES,
               'counts': dict(counts), 'versions': {}}
    draws = np.random.Generator(np.random.PCG64(BOOT_SEED)).integers(
        0, len(image_ids), size=(BOOT_RESAMPLES, len(image_ids)), dtype=np.int32)
    np.save(out_dir / ('task_c_bootstrap_draws_%s.npy' % tag), draws)
    for version in VERSIONS:
        print('TASK_C_EVALUATING', version, flush=True)
        evaluators = {}
        arm_stats = {}
        for arm in ARMS:
            coco_dt = coco.loadRes(dets[(version, arm)])
            evaluator = COCOeval(coco, coco_dt, 'bbox')
            evaluator.params.imgIds = image_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            evaluators[arm] = evaluator
            arm_stats[arm] = {'AP': float(evaluator.stats[0]), 'AP50': float(evaluator.stats[1]),
                              'AP75': float(evaluator.stats[2]), 'APs': float(evaluator.stats[3]),
                              'APm': float(evaluator.stats[4]), 'APl': float(evaluator.stats[5])}
            del coco_dt
            identity = np.arange(len(image_ids))
            recon_ap = ap_from_matches(evaluator, identity, area_index=0)
            recon_aps = ap_from_matches(evaluator, identity, area_index=1)
            recon_apl = ap_from_matches(evaluator, identity, area_index=3)
            print('TASK_C_RECONSTRUCTION', version, arm,
                  json.dumps({'AP_diff': recon_ap - arm_stats[arm]['AP'],
                              'APs_diff': recon_aps - arm_stats[arm]['APs'],
                              'APl_diff': recon_apl - arm_stats[arm]['APl']}, sort_keys=True), flush=True)
            assert abs(recon_ap - arm_stats[arm]['AP']) < 1e-8
            assert abs(recon_aps - arm_stats[arm]['APs']) < 1e-8
            assert abs(recon_apl - arm_stats[arm]['APl']) < 1e-8
        clean_stats = arm_stats['clean']
        deltas = {arm: {key: arm_stats[arm][key] - clean_stats[key] for key in clean_stats}
                  for arm in ARMS if arm != 'clean'}
        boot = {}
        if not args.skip_bootstrap:
            needed = ('clean',) + INSIDE_ARMS + OUTSIDE_ARMS
            boot_arrays = {}
            for arm in needed:
                evaluator = evaluators[arm]
                aps_values = []
                apl_values = []
                for draw in draws:
                    aps_values.append(ap_from_matches(evaluator, draw, area_index=1))
                    apl_values.append(ap_from_matches(evaluator, draw, area_index=3))
                boot_arrays[arm] = {'APs': np.asarray(aps_values), 'APl': np.asarray(apl_values)}
                np.save(out_dir / ('task_c_boot_%s_%s_%s.npy' % (tag, version, arm.replace('.', ''))),
                        np.stack([np.asarray(aps_values), np.asarray(apl_values)]))
            for arm in INSIDE_ARMS + OUTSIDE_ARMS:
                s_values = ((boot_arrays[arm]['APs'] - boot_arrays['clean']['APs']) -
                            (boot_arrays[arm]['APl'] - boot_arrays['clean']['APl']))
                observed = deltas[arm]['APs'] - deltas[arm]['APl']
                boot[arm] = {'S_observed': float(observed),
                             'S_bootstrap_mean': float(s_values.mean()),
                             'S_ci95': np.quantile(s_values, [.025, .975]).tolist()}
                print('TASK_C_BOOTSTRAP', args.model, version, arm,
                      json.dumps(boot[arm], sort_keys=True), flush=True)
        results['versions'][version] = {'arms': arm_stats, 'deltas_vs_clean': deltas,
                                        'S_bootstrap': boot}
    results['prop_dose_max_abs_error'] = {k: float(v) for k, v in dose_err.items()}
    print('Y_PROP_DOSE_MAX_ERR', json.dumps(results['prop_dose_max_abs_error'], sort_keys=True), flush=True)
    results['dose_accounting'] = {
        '%s/%s/layer%d' % key: {'n_queries': int(dose_cnt[key]),
                                'mean_removed_mass': float(dose_acc[key] / max(dose_cnt[key], 1))}
        for key in sorted(dose_acc)}
    out_path = out_dir / ('task_c_%s_summary.json' % tag)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
    print('TASK_C_SUMMARY', str(out_path), file_sha256(out_path), flush=True)


if __name__ == '__main__':
    main()
