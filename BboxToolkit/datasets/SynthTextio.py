import os
import time
import numpy as np
import scipy.io as scio
import os.path as osp

from functools import partial
from .misc import img_exts, prog_map, nproc_map
from ..imagesize import imsize
from ..geometry import bbox_areas


def load_synthtext(img_dir, ann_dir=None, classes=None, nproc=10, box_key='wordBB'):
    if classes is not None:
        print('load_synthtext loads all objects as `text`, arguments classes is no use')
    assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'
    assert box_key in ['wordBB', 'charBB']

    print('Starting loading SynthText dataset information.')
    start_time = time.time()
    if ann_dir is not None:
        assert osp.isfile(ann_dir) and ann_dir.endswith('.mat'), \
                f'Invaild ann_dir for SynthText {ann_dir}'
        data = scio.loadmat(ann_dir)
        bboxes = data[box_key][0]
        imnames = data['imnames'][0]
        _contents = nproc_map(_parse_synthtext_mat,
                              list(zip(imnames, bboxes)),
                              nproc=nproc)
    else:
        _contents = []
        for root, _, files in os.walk(img_dir):
            root = root.replace(img_dir, '', 1)
            for f in files:
                if osp.splitext(f)[-1] not in img_exts:
                    continue
                filename = osp.join(root, f)
                filename.lstrip('/')

                bboxes = np.zeros((0, 8), dtype=np.float32)
                labels = np.zeros((0, ), dtype=np.int64)
                ann = dict(bboxes=bboxes, labels=labels)
                _contents.append(dict(filename=filename, ann=ann))

    _load_func = partial(_merge_img_size, root=img_dir)
    contents = prog_map(_load_func, _contents, nproc)
    end_time = time.time()
    print(f'Finishing loading SynthText, get {len(contents)} images, ',
          f'using {end_time-start_time:.3f}s.')
    return contents, ('text', )


def _merge_img_size(content, root):
    imgfile = osp.join(root, content['filename'])
    if not osp.exists(imgfile):
        print('image {imgfile} is not exists, discard this image information')
        return None

    width, height = imsize(imgfile)
    content.update(dict(width=width, height=height))

    bboxes = content['ann']['bboxes']
    min_coors = bboxes.min(axis=1)
    max_x = bboxes[:, 0::2].max(axis=1)
    max_y = bboxes[:, 1::2].max(axis=1)
    error = (min_coors < 0) | (max_x > width) | (max_y > height)
    if error.any():
        bboxes = bboxes[~error]
        labels = content['ann']['labels'][~error]
        content['ann']['bboxes'] = bboxes
        content['ann']['labels'] = labels
    return content


def _parse_synthtext_mat(info):
    imgfile, bboxes = info
    bboxes = bboxes.astype(np.float32)
    if bboxes.ndim == 2:
        bboxes = bboxes[..., None]
    bboxes = bboxes.transpose(2, 1, 0)

    pt_a, pt_b, pt_c, pt_d = np.split(bboxes, 4, axis=1)
    cross1 = np.cross(pt_a-pt_d, pt_c-pt_d) * np.cross(pt_b-pt_d, pt_c-pt_d)
    cross2 = np.cross(pt_c-pt_b, pt_a-pt_b) * np.cross(pt_d-pt_b, pt_a-pt_b)
    cross = (cross1 < 0) & (cross2 < 0)
    cross = cross.squeeze(-1)
    if cross.any():
        bboxes[cross] = bboxes[cross][:, [0, 2, 1, 3], :]

    labels = np.zeros((len(bboxes), ), dtype=np.int64)
    ann = dict(bboxes=bboxes, labels=labels)
    content = dict(filename=imgfile[0], ann=ann)
    return content
