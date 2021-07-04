import os
import numpy as np
import os.path as osp

from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from .base import register_io_func
from ..utils import img_exts, imgsize
from ..structures import P4POLY


@register_io_func('load', 'dota')
def load_dota(img_dir, img_ext=None, ann_dir=None, ann_ext='.txt', nproc=1):
    '''Loading data formatted as DOTA dataset.

    Args:
        img_dir (str): Path to images.
        img_ext (None | str): The suffix of images. If img_ext is None, The
            function will regard image extentions as suffix.
        ann_dir (str): Path to annotations.
        ann_ext (str): The suffic of annotations.
        nproc (int): number of processions.

    Returns:
        loading function outputs
        see: func: `BboxToolkit.datasets.load_dataset`
    '''

    # Single function for multiprocesing.
    def _single_func(imgfile):
        # print(img
        if img_ext is None:
            img_id, ext = osp.splitext(imgfile)
            if ext not in img_exts:
                return None
        else:
            if not imgfile.endswith(img_ext):
                return None
            img_id = imgfile[:-len(img_ext)]

        if ann_dir is None:
            content = dict(gsd=None, ann=dict())
        else:
            txtfile = osp.join(ann_dir, img_id+ann_ext)
            content = read_dota_txt(txtfile)

        imgpath = osp.join(img_dir, imgfile)
        size = imgsize(imgpath)
        content.update(dict(filename=imgfile, width=size[0], height=size[1]))
        return content

    if nproc > 1:
        # Multiprocessing
        pool = ProcessPool(min(nproc, os.cpu_count()))
        contents = pool.map(_single_func, os.listdir(img_dir))
        pool.close()
    else:
        # Singleprocessing
        contents = []
        for imgfile in tqdm(os.listdir(img_dir)):
            contents.append(_single_func(imgfile))

    contents = [c for c in contents if c is not None]
    return contents


def read_dota_txt(txtfile):
    '''Read the DOTA txt annotation file.'''
    gsd, bboxes, diffs, categories = None, [], [], []
    with open(txtfile, 'r') as f:
        for line in f:
            items = line.split(' ')
            if len(items) >= 9:
                bboxes.append([float(i) for i in items[:8]])
                diffs.append(int(items[9]) if len(items) == 10 else 0)
                categories.append(items[8])
                continue

            if line.startswith('gsd'):
                try:
                    gsd = float(line.split(':')[-1])
                except ValueError:
                    pass

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
            np.zeros((0, 8), dtype=np.float32)
    diffs = np.array(diffs, dtype=np.int64) if diffs else \
            np.zeros((0, ), dtype=np.int64)
    ann = dict(bboxes=P4POLY(bboxes), diffs=diffs, categories=categories)
    return dict(gsd=gsd, ann=ann)
