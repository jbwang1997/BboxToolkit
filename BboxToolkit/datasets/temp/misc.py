import os
import os.path as osp

from ..utils import img_exts


def read_imgset(img_set):
    if isinstance(img_set, str):
        assert osp.isfile(img_set)
        with open(img_set, 'r') as f:
            img_set = f.readlines()

    assert isinstance(img_set, list)
    id_list = []
    for line in img_set:
        img_id, ext = osp.splitext(line.strip())
        if ext:
            assert ext in img_exts
        id_list.append(img_id)
    return id_list


def read_imgdir(img_dir):
    id2img = dict()
    for f in os.listdir(img_dir):
        img_id, ext = osp.splitext(f)
        if ext in img_exts:
            id2img[img_id] = f
    return id2img
