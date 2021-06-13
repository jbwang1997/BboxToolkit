import os
import time
import pickle
import os.path as osp
import BboxToolkit.datasets as datasets

from tqdm import tqdm
from .class_names import get_classes
from .misc import change_cls_order, split_imgset
from ..utils import img_exts, print_log, imgsize


#
# BboxToolkit.datasets provides some dataset IOs out of box.
# If you want to add a new dataset IO, you need to follow some rules to
# make your functions compatible with other parts of BboxToolkits.

# Loading Functions:
#    name: load_xxx
#    args:
#       img_dir(str): The path to images
# 


def load_dataset(load_type, **kwargs):
    assert load_type != 'dataset'
    logger = kwargs.get('logger', None)
    print_log(f'Start loading {load_type} dataset.', logger)
    for k, v in kwargs.items():
        if k == 'logger':
            continue
        print_log(f'{k}: {v}', logger)

    start_time = time.time()
    load_func = getattr(datasets, 'load_'+load_type)
    data, classes = load_func(**kwargs)
    end_time = time.time()

    print_log(f'Finish loading {load_type} dataset.', logger)
    print_log(f'Time consuming: {end_time - start_time:.3f}s.', logger)
    print_log(f'Data number: {len(data)}', logger)
    return data, classes


def load_imgs(img_dir, img_set=None, classes=None, **kwargs):
    id2img = dict()
    for f in os.listdir(img_dir):
        img_id, ext = osp.splitext(f)
        if ext in img_exts:
            id2img[img_id] = f

    if img_set is None:
        id_list = list(id2img.keys())
    elif isinstance(img_set, list):
        id_list = [osp.splitext(f)[0] for f in img_set]
    elif isinstance(img_set, str):
        id_list = []
        with open(img_set, 'r') as f:
            for line in f:
                line = line.strip()
                id_list.append(osp.splitext(line)[0])
    else:
        raise TypeError('img_set should be None, list or str, ',
                        f'but get {type(img_set)}')

    data = []
    for img_id in tqdm(id_list):
        assert img_id in id2img, f'Cannot find{img_id} in {img_dir}'
        imgfile = id2img[img_id]
        width, height = imgsize(osp.join(img_dir, imgfile))
        data.append(dict(
            width=width,
            height=height,
            filename=imgfile,
            ann=dict()
        ))

    classes = [] if classes is None else get_classes(classes)
    return data, classes


def load_pkl(ann_dir, img_set=None, classes=None, **kwargs):
    with open(ann_dir, 'rb') as f:
        pkl_loading = pickle.load(f)
    old_classes = pkl_loading['classes']
    data = pkl_loading['data']

    if img_set is not None:
        data = split_imgset(data, img_set)
    if classes is not None:
        change_cls_order(data, old_classes, classes)

    return data, classes


def dump_dataset(dump_type, **kwargs):
    assert dump_type != 'dataset'
    logger = kwargs.get('logger', None)
    print_log(f'Start dumping {dump_type} dataset.', logger)
    for k, v in kwargs.items():
        if k == 'logger':
            continue
        print_log(f'{k}: {v}', logger)

    start_time = time.time()
    dump_func = getattr(datasets, 'dump_'+dump_type)
    dump_func(**kwargs)
    end_time = time.time()

    print_log(f'Finish dumping data as {dump_type} dataset.')
    print_log(f'Time consuming: {end_time - start_time:.3f}s.', logger)


def dump_pkl(ann_dir, data, classes):
    assert ann_dir.endswith('.pkl')
    filepath = osp.split(ann_dir)[0]
    if not osp.exists(filepath):
        os.makedirs(filepath)

    data = dict(classes=classes, data=data)
    with open(ann_dir, 'wb') as f:
        pickle.dump(data, f)
