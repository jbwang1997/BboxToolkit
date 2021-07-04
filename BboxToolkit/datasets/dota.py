# import re
# import os
# import time
# import warnings

# import os.path as osp
# import numpy as np

# from PIL import Image
# from functools import reduce, partial
# from multiprocessing import Pool
# from collections import defaultdict

# from .io import load_imgs
# from .misc import get_classes, img_exts
# from ..utils import get_bbox_type
# from ..geometry import bbox2type

from .base import register_io_func
from ..utils import img_exts, imgsize


@register_io_func('load', 'dota')
def load_dota(img_dir, ann_dir=None):
    pass
    



def load_dota(img_dir, ann_dir=None, classes=None, nproc=10):
    classes = get_classes('DOTA' if classes is None else classes)
    cls2lbl = {cls: i for i, cls in enumerate(classes)}

    print('Starting loading DOTA dataset information.')
    start_time = time.time()
    _load_func = partial(_load_dota_single,
                         img_dir=img_dir,
                         ann_dir=ann_dir,
                         cls2lbl=cls2lbl)
    if nproc > 1:
        pool = Pool(nproc)
        contents = pool.map(_load_func, os.listdir(img_dir))
        pool.close()
    else:
        contents = list(map(_load_func, os.listdir(img_dir)))
    contents = [c for c in contents if c is not None]
    end_time = time.time()
    print(f'Finishing loading DOTA, get {len(contents)} images,',
          f'using {end_time-start_time:.3f}s.')

    return contents, classes
