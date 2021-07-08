import os
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET

from tqdm import tqdm
from .misc import get_id2img, read_imgset
from .class_names import voc_classes
from ..structures import HBB
from ..utils import imgsize


def load_voc(img_dir, img_set=None, ann_dir=None, logger=None, **kwargs):
    classes = voc_classes()
    cls2lbl = {cls: i for i, cls in enumerate(classes)}

    id2img = get_id2img(img_dir)
    if img_set is None:
        id_list = list(id2img.keys())
    else:
        id_list = read_imgset(img_set)

    data = []
    for img_id in id_list:
        assert img_id in id2img, f'Cannot find{img_id} in {img_dir}'
        imgfile = id2img[img_id]

    



def _read_voc_xml():
    pass


def dump_voc():
    pass
