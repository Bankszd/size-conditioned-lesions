"""Task R: proportional-to-clean-mass lesions (f of the anchor's own in-box mass).

COCO dev branch identical to task_m.py (all equal-mass arms kept), plus VOC 2007 test branch.
Arms: fixed_inside, mass_<m> for every m in cfg["mass_targets"], prop_<f> for every f in
cfg["relative_targets"].
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
import hashlib
import json
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

from g3 import area_bin, choose_anchors, crowd_count, sample_cells
from intervene import (Controller, analytic_sanity, centers_and_mask, ground_truth,
                       load_model, local_quality, take_mass, utility)
from mve import file_sha256

VOC_COCO_IDS = (1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21,
                44, 62, 63, 64, 67, 72)  # 20 VOC classes as COCO category ids


def raw_dets(output, model_name):
    logits = output['pred_logits'][0]
    boxes = output['pred_boxes'][0]
    if model_name == 'conditional_detr':
        probs = logits.sigmoid()
    else:
        probs = logits.softmax(-1)[..., :-1]
    scores, labels = probs.max(-1)
    return scores, labels, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--model', choices=['detr', 'conditional_detr'], required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--identity-images', type=int, default=20)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--data', choices=['coco', 'voc'], default='coco')
    ap.add_argument('--voc-json', default='_rel(DATA_ROOT, "")/voc2007test/instances_voc2007test.json')
    ap.add_argument('--voc-images', default='_rel(DATA_ROOT, "")/VOCdevkit/VOC2007/JPEGImages')
    args = ap.parse_args()

    cfg = json.loads(pathlib.Path(args.config).read_text())
    assert cfg['mode'] == 'task_r'
    assert cfg['random_seeds'] == [20270930]
    torch.set_num_threads(4)
    torch.manual_seed(20270930)
    torch.backends.cudnn.benchmark = False

    model, post, module, transform = load_model(args.model, args.cache)
    print('TASKR_CHECKPOINT_SHA256', args.model,
          file_sha256(pathlib.Path(args.cache) / (args.model + '.pth')), flush=True)
    assert analytic_sanity() < 1e-6
    controller = Controller(module)
    holder = {}

    def feature_shape(_m, _i, output):
        holder['shape'] = tuple(output[0][-1].tensors.shape[-2:])

    model.backbone.register_forward_hook(feature_shape)
    matcher_module = __import__('models.matcher', fromlist=['HungarianMatcher'])
    HungarianMatcher = matcher_module.HungarianMatcher
    matcher = HungarianMatcher(cost_class=1 if args.model == 'detr' else 2,
                               cost_bbox=5, cost_giou=2)
    hm_voc = None
    if args.data == 'voc':
        try:
            hm_voc = HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                                      use_focal_loss=True, alpha=0.25, gamma=2.0)
        except TypeError:
            hm_voc = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

    if args.data == 'voc':
        coco = COCO(args.voc_json)
        vocab_ids = sorted(int(x) for x in coco.getImgIds())
        digest = hashlib.sha256(','.join(str(x) for x in vocab_ids).encode()).hexdigest()
        print('TASKR_VOC_IDS_SHA256', digest, 'N', len(vocab_ids), flush=True)
        image_ids = vocab_ids[args.offset:]
        voc_ids = torch.tensor(VOC_COCO_IDS, device='cuda', dtype=torch.long)
    else:
        assert cfg['sample_offset'] == 1000 and cfg['sample_size'] == 4000
        coco = COCO(str(pathlib.Path(args.cache) / 'coco/annotations/instances_val2017.json'))
        perm = np.random.Generator(np.random.PCG64(cfg['sample_seed'])).permutation(sorted(coco.getImgIds()))
        image_ids = sorted(int(x) for x in perm[1000:5000])
        digest = hashlib.sha256(','.join(str(x) for x in image_ids).encode()).hexdigest()
        assert digest == cfg['holdout_sha256'], (digest, cfg['holdout_sha256'])
        print('TASKR_HOLDOUT_IDS_SHA256', digest, 'N', len(image_ids), flush=True)
        voc_ids = None
    if args.limit:
        image_ids = image_ids[:args.limit]
        print('TASKR_SMOKE_LIMIT', args.limit, flush=True)

    outpath = pathlib.Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    ident = {'n_images': 0, 'logit_maxabs': 0.0, 'box_maxabs': 0.0}
    nontrivial = []
    start = time.time()

    with open(outpath, 'w', encoding='utf8') as raw, torch.no_grad():
        for ordinal, image_id in enumerate(image_ids, 1):
            info = coco.loadImgs([image_id])[0]
            img_path = (os.path.join(args.voc_images, info['file_name']) if args.data == 'voc'
                        else str(pathlib.Path(args.cache) / 'coco/val2017' / info['file_name']))
            image = Image.open(img_path).convert('RGB')
            tensor, _ = transform(image, None)
            tensor = tensor.cuda()
            clean = controller.forward(model, tensor)
            attention = controller.weights.copy()
            shape = holder['shape']
            if args.data == 'voc':
                size = (int(info['height']), int(info['width']))
            else:
                size = (image.height, image.width)

            if ordinal <= args.identity_images:
                noop = controller.forward(model, tensor, [], 'all')
                ld = float((noop['pred_logits'] - clean['pred_logits']).abs().max())
                bd = float((noop['pred_boxes'] - clean['pred_boxes']).abs().max())
                ident['logit_maxabs'] = max(ident['logit_maxabs'], ld)
                ident['box_maxabs'] = max(ident['box_maxabs'], bd)
                ident['n_images'] += 1

            anns, target = ground_truth(coco, image_id, size[1], size[0])
            if args.data == 'voc':
                _sc, labels_all, _bx = raw_dets(clean, args.model)
                keep = torch.nonzero(torch.isin(labels_all, voc_ids), as_tuple=False).flatten()
                sub_out = {'pred_logits': clean['pred_logits'][:, keep],
                           'pred_boxes': clean['pred_boxes'][:, keep]}
                idx = hm_voc(sub_out, [target])[0]
                pairs = [(int(keep[q]), int(g)) for q, g in zip(idx[0].tolist(), idx[1].tolist())]
                best = {}
                for q, gi in pairs:
                    ann = anns[gi]
                    b = area_bin(float(ann['area']))
                    if b not in best or ann['id'] < best[b][1]['id']:
                        best[b] = (q, ann)
                anchors = [best[b] for b in ('small', 'medium', 'large') if b in best]
                counts['matched_pairs'] += len(pairs)
            else:
                anchors = choose_anchors(matcher, clean, anns, target)
            counts['images'] += 1
            for query, ann in anchors:
                size_bin = area_bin(ann['area'])
                counts['candidate_' + size_bin] += 1
                x, y, bw, bh = ann['bbox']
                box = np.array([x, y, x + bw, y + bh])
                _, _, in_mask = centers_and_mask(shape, size, box)
                in_ids = np.flatnonzero(in_mask)
                out_ids = np.flatnonzero(~in_mask)
                if len(in_ids) < 4 or len(out_ids) < 4:
                    counts['excluded_overlap_' + size_bin] += 1
                    continue
                weights = attention[query]
                clean_local = local_quality(clean, post, query, ann, size, args.model)
                base = 20270930 + image_id * 1009 + ann['id'] * 7919
                inside = sample_cells(in_ids, 4, base)
                mass_eligible = float(weights[in_ids].sum())

                arms = [('fixed_inside', [(j, 1.0) for j in inside], None, 'inside')]

                shuffled_in = np.random.Generator(np.random.PCG64(int(base + 2))).permutation(in_ids)
                for mass in cfg['mass_targets']:
                    name = 'mass_%g' % mass
                    if mass_eligible + 1e-7 < mass:
                        counts['excluded_' + name + '_' + size_bin] += 1
                        continue
                    selected = take_mass(shuffled_in, weights, float(mass))
                    assert selected is not None
                    achieved = sum(float(weights[j]) * f for j, f in selected)
                    assert abs(achieved - mass) <= 1e-4
                    arms.append((name, selected, float(mass), 'inside'))

                for frac in cfg['relative_targets']:
                    target_mass = float(frac) * mass_eligible
                    sel = take_mass(shuffled_in, weights, target_mass)
                    if sel is None:
                        counts['excluded_prop_%g_%s' % (frac, size_bin)] += 1
                        continue
                    got = sum(float(weights[j]) * f for j, f in sel)
                    assert abs(got - target_mass) <= 1e-4, (frac, got, target_mass)
                    arms.append(('prop_%g' % frac, sel, target_mass, 'inside'))

                if ordinal <= args.identity_images and len(nontrivial) < 5:
                    full_cells = [(int(j), 1.0) for j in in_ids]
                    pert = controller.forward(model, tensor, full_cells, 'all', query)
                    q_full = local_quality(pert, post, query, ann, size, args.model)
                    nontrivial.append({'image_id': int(image_id), 'annotation_id': int(ann['id']),
                                       'clean_local': clean_local, 'full_delete_local': q_full,
                                       'n_cells': len(full_cells)})

                for arm_name, cells, nominal, side in arms:
                    removed = sum(float(weights[j]) * f for j, f in cells)
                    assert len(cells) < attention.shape[1]
                    pert = controller.forward(model, tensor, cells, 'all', query)
                    q = local_quality(pert, post, query, ann, size, args.model)
                    row = {'model': args.model, 'image_id': int(image_id),
                           'annotation_id': int(ann['id']), 'query': int(query),
                           'area': ann['area'], 'area_bin': size_bin,
                           'layer_index_zero_based': 2, 'mode': 'all', 'seed': 20270930,
                           'arm': arm_name, 'side': side, 'nominal_mass': nominal,
                           'removed_attention_mass': removed,
                           'clean_in_box_attention_mass': mass_eligible,
                           'attenuated_cell_count': len(cells),
                           'fractional_cell_count': sum(0 < f < 1 for _, f in cells),
                           'box_cells': len(in_ids), 'outside_cells': len(out_ids),
                           'Q_clean': clean_local,
                           'Q_inside': q if side == 'inside' else None,
                           'Q_outside': q if side == 'outside' else None,
                           'perturbed_local': q}
                    if arm_name.startswith('prop_'):
                        row['relative_target'] = float(arm_name.split('_')[1])
                    raw.write(json.dumps(row, sort_keys=True) + '\n')
                    counts['rows_' + arm_name + '_' + size_bin] += 1
            if ordinal % 100 == 0:
                raw.flush()
                print('TASKR_PROGRESS', args.model, ordinal, 'elapsed', round(time.time() - start, 1), flush=True)

    sha = file_sha256(outpath)
    summary = {'model': args.model, 'data': args.data, 'counts': dict(counts),
               'elapsed_s': round(time.time() - start, 1), 'ids_sha256': digest,
               'raw_path': str(outpath), 'raw_sha256': sha, 'identity': ident,
               'nontrivial_full_delete': nontrivial, 'config': cfg}
    print('TASKR_DONE', json.dumps({'summary': summary}, sort_keys=True), flush=True)
    pathlib.Path(str(outpath) + '.summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
