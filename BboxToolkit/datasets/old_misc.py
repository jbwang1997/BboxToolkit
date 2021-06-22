import os
import numpy as np
import os.path as osp

from ..utils import img_exts


def change_cls_order(data, old_classes, new_classes):
    for n_c, o_c in zip(new_classes, old_classes):
        if n_c != o_c:
            break
    else:
        if len(old_classes) == len(new_classes):
            return

    new_cls2lbl = {cls: i for i, cls in enumerate(new_classes)}
    lbl_mapper = [new_cls2lbl[cls] if cls in new_cls2lbl else -1
                  for cls in old_classes]
    lbl_mapper = np.array(lbl_mapper, dtype=np.int64)

    for d in data:
        if 'labels' not in d['ann']:
            continue

        new_labels = lbl_mapper[d['ann']['labels']]
        d['ann']['labels'] = new_labels

        if (new_labels == -1).any():
            assert new_labels.ndim == 1
            inds = np.nonzero(new_labels != -1)[0]
            for k, v in d['ann'].items():
                try:
                    d['ann'][k] = v[inds]
                except TypeError:
                    d['ann'][k] = type(v)([v[i] for i in inds])


def get_id2img(img_dir):
    id2img = dict()
    for f in os.listdir(img_dir):
        img_id, ext = osp.splitext(f)
        if ext in img_exts:
            id2img[img_id] = f
    return id2img


def read_imgset(img_set):
    if isinstance(img_set, list):
        return [osp.splitext(f)[0] for f in img_set]
    elif isinstance(img_set, str):
        assert osp.isfile(img_set)
        id_list = []
        with open(img_set, 'r') as f:
            for line in f:
                line = line.strip()
                id_list.append(osp.splitext(line)[0])
        return id_list
    else:
        raise TypeError('img_set should be None, list or str, ',
                        f'but get {type(img_set)}')


def split_imgset(data, img_set):
    new_data = []
    img_set = read_imgset(img_set)
    id_mapper = {osp.splitext(d['filename'])[0]: d for d in data}
    for img_id in img_set:
        new_data.append(id_mapper[img_id])
    return new_data
