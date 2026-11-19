# -*- coding: utf-8 -*-
"""Build the VOC 2012 val annotations in the same COCO layout as the VOC 2007 test file.

This is mk_voc_coco_local.py with three substitutions only: ROOT, OUT and the split file
(test.txt -> val.txt), plus the output file name. The class map, the integer image ids
(sorted name order, 1-based), the difficult -> iscrowd rule, the area = w*h rule and the
bbox convention are identical.
"""
import json
import os, xml.etree.ElementTree as ET

ROOT = os.environ.get('VOC2012_ROOT', 'data/VOCdevkit/VOC2012')
OUT = os.environ.get('OUT_ROOT', 'data/voc2012_coco')
os.makedirs(OUT, exist_ok=True)

COCO_MAP = {'aeroplane':'airplane','bicycle':'bicycle','bird':'bird','boat':'boat','bottle':'bottle',
 'bus':'bus','car':'car','cat':'cat','chair':'chair','cow':'cow','diningtable':'dining table',
 'dog':'dog','horse':'horse','motorbike':'motorcycle','person':'person','pottedplant':'potted plant',
 'sheep':'sheep','sofa':'couch','train':'train','tvmonitor':'tv'}
COCO_ID = {'person':1,'bicycle':2,'car':3,'motorcycle':4,'airplane':5,'bus':6,'train':7,'truck':8,
 'boat':9,'bird':16,'cat':17,'dog':18,'horse':19,'sheep':20,'cow':21,'bottle':44,'chair':62,
 'couch':63,'potted plant':64,'dining table':67,'tv':72}

names = [l.strip() for l in open(os.path.join(ROOT,'ImageSets/Main/val.txt')) if l.strip()]
print('images in val.txt', len(names))
images, anns = [], []
next_ann = 1
for idx, name in enumerate(sorted(names)):
    x = ET.parse(os.path.join(ROOT,'Annotations', name + '.xml')).getroot()
    w, h = int(x.find('size/width').text), int(x.find('size/height').text)
    images.append({'id': idx+1, 'file_name': name + '.jpg', 'width': w, 'height': h})
    for order, obj in enumerate(x.findall('object'), 1):
        cls = COCO_MAP[obj.find('name').text.strip()]
        bb = obj.find('bndbox')
        x1 = float(bb.find('xmin').text); y1 = float(bb.find('ymin').text)
        x2 = float(bb.find('xmax').text); y2 = float(bb.find('ymax').text)
        bw, bh = max(x2-x1, 0.0), max(y2-y1, 0.0)
        diffic = int(obj.find('difficult').text)
        anns.append({'id': next_ann, 'image_id': idx+1, 'category_id': COCO_ID[cls],
                     'bbox': [x1, y1, bw, bh], 'area': bw*bh,
                     'iscrowd': 1 if diffic else 0, 'voc_name': name, 'voc_order': order})
        next_ann += 1

cats = [{'id': v, 'name': k} for k, v in COCO_ID.items()]
json.dump({'info': {}, 'licenses': [], 'images': images, 'annotations': anns, 'categories': cats},
          open(os.path.join(OUT,'instances_voc2012val.json'),'w'))

bins = {'small': 0, 'medium': 0, 'large': 0}
for a in anns:
    if a['iscrowd']: continue
    ar = a['area']
    b = 'small' if ar < 32**2 else ('medium' if ar < 96**2 else 'large')
    bins[b] += 1
print('non-difficult by bin', bins, 'total', sum(bins.values()))

per = {'small':0,'medium':0,'large':0}
by_img = {}
for a in anns:
    if a['iscrowd']: continue
    ar = a['area']
    b = 'small' if ar < 32**2 else ('medium' if ar < 96**2 else 'large')
    by_img.setdefault(a['image_id'], set()).add(b)
for s in by_img.values():
    for b in s: per[b] += 1
print('candidate anchors (<=1 GT per image per bin)', per)
print('images', len(images), 'anns', len(anns))
