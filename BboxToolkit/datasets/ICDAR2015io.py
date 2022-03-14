import os
import time
import numpy as np
import os.path as osp

from functools import partial
from .misc import img_exts, prog_map
from ..imagesize import imsize


def load_icdar2015(img_dir, ann_dir=None, classes=None, nproc=10):
    if classes is not None:
        print('load_icdar2015 loads all objects as `text`, arguments classes is no use')
    assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'
    assert ann_dir is None or osp.isdir(ann_dir), f'The {ann_dir} is not an existing dir!'

    imgpaths = [f for f in os.listdir(img_dir) if f[-4:] in img_exts]
    _load_func = partial(_load_icdar2015_single,
                         img_dir=img_dir,
                         ann_dir=ann_dir)

    print('Starting loading ICDAR2015 dataset information.')
    start_time = time.time()
    contents = prog_map(_load_func, imgpaths, nproc)
    end_time = time.time()
    print(f'Finishing loading ICDAR2015, get {len(contents)} images, ',
          f'using {end_time-start_time:.3f}s.')
    return contents, ['text']


def _load_icdar2015_single(imgfile, img_dir, ann_dir):
    img_id, _ = osp.splitext(imgfile)
    txtfile = None if ann_dir is None else \
            osp.join(ann_dir, 'gt_'+img_id+'.txt')
    content = _load_icdar2015_txt(txtfile)

    imgfile = osp.join(img_dir, imgfile)
    width, height = imsize(imgfile)
    content.update(dict(width=width, height=height, filename=imgfile, id=img_id))
    return content


def _load_icdar2015_txt(txtfile):
    bboxes, texts = [], []
    if txtfile is None:
        pass
    elif not osp.isfile(txtfile):
        print(f'Cannot find {txtfile}, treated as empty txt')
    else:
        with open(txtfile, 'r', encoding='utf-8-sig') as f:
            for line in f:
                items = line.strip().split(',')
                bboxes.append([int(i) for i in items[:8]])
                texts.append(items[8])


    bboxes = np.array(bboxes, dtype=np.float32) if bboxes \
            else np.zeros((0, 8), dtype=np.float32)
    labels = np.zeros((bboxes.shape[0], ), dtype=np.int64)
    ann = dict(bboxes=bboxes, labels=labels, texts=texts)
    return dict(ann=ann)
