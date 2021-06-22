import os
import time
import pickle
import os.path as osp
import BboxToolkit.datasets as datasets

from tqdm import tqdm
from .misc import read_imgset, read_imgdir
from ..utils import print_log, imgsize


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
    data = load_func(**kwargs)
    end_time = time.time()

    print_log(f'Finish loading {load_type} dataset.', logger)
    print_log(f'Time consuming: {end_time - start_time:.3f}s.', logger)
    print_log(f'Data number: {len(data)}', logger)
    return data


def load_imgs(img_dir, img_set=None, **kwargs):
    id2img = read_imgdir(img_dir)
    imgset = list(id2img) if img_set is None \
            else read_imgset(img_set)

    data = []
    for img_id in tqdm(imgset):
        assert img_id in id2img, f'Cannot find{img_id} in {img_dir}'
        imgfile = id2img[img_id]
        width, height = imgsize(osp.join(img_dir, imgfile))
        data.append(dict(
            width=width, height=height, filename=imgfile, ann=dict()))
    return data


def load_pkl(ann_dir, img_set=None, **kwargs):
    with open(ann_dir, 'rb') as f:
        data = pickle.load(f)

    if img_set is not None:
        data = split_imgset(data, img_set)
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


def dump_pkl(ann_dir, data, classes, **kwargs):
    assert ann_dir.endswith('.pkl')
    filepath = osp.split(ann_dir)[0]
    if not osp.exists(filepath):
        os.makedirs(filepath)

    data = dict(classes=classes, data=data)
    with open(ann_dir, 'wb') as f:
        pickle.dump(data, f)
