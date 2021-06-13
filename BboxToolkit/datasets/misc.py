import os.path as osp
import numpy as np


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


def split_imgset(data, img_set):
    if isinstance(img_set, str):
        with open(img_set, 'r') as f:
            img_set = [s.strip() for s in f]
    else:
        assert isinstance(img_set, list)

    new_data = []
    id_mapper = {osp.splitext(d['filename'])[0]: d for d in data}
    for img_id in img_set:
        img_id = osp.splitext(img_id)[0]
        new_data.append(id_mapper[img_id])

    return new_data
