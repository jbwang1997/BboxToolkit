import os
import cv2
import copy
import time
import pickle
import logging
import argparse
import datetime
import numpy as np
import os.path as osp

from math import ceil
from functools import partial, reduce
from itertools import product, count
from multiprocessing import Pool, Manager
from cloghandler import ConcurrentRotatingFileHandler

from BboxToolkit.utils import Config
from BboxToolkit.ops import bbox_overlaps
from BboxToolkit.datasets import load_dataset
from BboxToolkit.structures import BaseBbox, HBB


def slide_windows(W, H, ws, hs, xs, ys, iof_thr=0.6):
    assert len(ws) == len(hs) == len(xs) == len(ys)

    windows = []
    for w, h, x, y in zip(ws, hs, xs, ys):
        assert (w >= x) and (h >= y), 'Invalid windows parameters ' + \
                f'w: {w}, x: {x}, h: {h}, y: {y}.'

        x_num = 1 if W <= w else ceil((W - w) / x + 1)
        x_start = [x * i for i in range(x_num)]
        if (len(x_start) > 1) and (x_start[-1] + w > W):
            x_start[-1] = W - w

        y_num = 1 if H <= h else ceil((H - h) / y + 1)
        y_start = [y * i for i in range(y_num)]
        if (len(y_start) > 1) and (y_start[-1] + h > H):
            y_start[-1] = H - h

        start = np.asarray(list(product(x_start, y_start)),
                           dtype=np.int64)
        stop = start + np.asarray([w, h], dtype=np.int64)
        windows.append(np.concatenate([start, stop], axis=1))

    windows = HBB(np.concatenate(windows, axis=0))
    image = HBB(np.asarray([[0, 0, W, H]]))
    win_iofs = bbox_overlaps(windows, image, mode='iof').reshape(-1)
    if not np.any(win_iofs >= iof_thr):
        win_iofs[abs(win_iofs - win_iofs.max()) < 0.01] = 1
    return windows[win_iofs >= iof_thr]


def crop_objects(content, windows, iof_thr, skip_empty, img_ext):
    bboxes = content['ann'].get('bboxes', HBB.gen_empty())
    iofs = bbox_overlaps(bboxes, windows, mode='iof')
    counter = count()
    win_contents = []
    for i in range(len(windows)):
        win_iofs = iofs[:, i]
        retain_idx = np.nonzero(win_iofs >= iof_thr)[0]
        if len(retain_idx) == 0 and skip_empty:
            continue

        win_content = dict()
        for k, v in content.items():
            if k in ['width', 'height', 'ann']:
                continue

            if k == 'filename':
                img_id, _ = osp.splitext(v)
                win_content['ori_filename'] = v
                win_content['filename'] = img_id + \
                        f'_{next(counter):04d}' + img_ext
            else:
                win_content[k] = copy.deepcopy(v)

        x_start, y_start, x_stop, y_stop = windows.bboxes[i]
        win_content['x_start'] = int(x_start)
        win_content['y_start'] = int(y_start)
        win_content['width'] = int(x_stop - x_start)
        win_content['height'] = int(y_stop - y_start)

        win_ann = dict()
        for k, v in content['ann'].items():
            if isinstance(v, BaseBbox):
                win_ann[k] = v[retain_idx].translate(-x_start, -y_start)
            else:
                try:
                    win_ann[k] = v[retain_idx]
                except TypeError:
                    win_ann[k] = [v[i] for i in retain_idx]
        win_ann['iofs'] = win_iofs[retain_idx]
        win_content['ann'] = win_ann
        win_contents.append(win_content)
    return win_contents


def crop_and_save_img(content, win_contents, padding_value, save_dir):
    if len(win_contents) == 0:
        return

    img = cv2.imread(osp.join(content['img_dir'], content['filename']))
    img_h, img_w, channels = img.shape
    for win_content in win_contents:
        filename = win_content['filename']
        x_start = win_content['x_start']
        y_start = win_content['y_start']
        width = win_content['width']
        height = win_content['height']

        if x_start + width > img_w or y_start + height > img_h:
            patch = np.full((height, width, channels), padding_value,
                            dtype=np.uint8)
            patch[:img_h - y_start, :img_w - x_start] = \
                    img[y_start:y_start + height,
                        x_start:x_start + width]
        else:
            patch = img[y_start:y_start + height,
                        x_start:x_start + width]
        cv2.imwrite(osp.join(save_dir, filename), patch)


def single_process(content, args, lock, prog, total, logger):
    W, H = content['width'], content['height']
    windows = slide_windows(
        W, H, args.ws, args.hs, args.xs, args.ys, args.img_iof_thr)
    patch_infos = crop_objects(content, windows, args.obj_iof_thr,
                               args.skip_empty, args.patch_ext)
    crop_and_save_img(content, patch_infos, args.padding_value,
                      args.dumping_img_dir)

    lock.acquire()
    prog.value += 1
    obj_num = len(content['ann'].get('bboxes', []))
    msg = f'({prog.value/total:3.1%} {prog.value}:{total})'
    msg += ' - ' + f"Filename: {content['filename']}"
    msg += ' - ' + f'W: {W:<5d}'
    msg += ' - ' + f'H: {H:<5d}'
    msg += ' - ' + f'Skip Empty: {str(args.skip_empty):<5s}'
    msg += ' - ' + f'Object Number: {obj_num:<4d}'
    msg += ' - ' + f'Windows: {len(windows):<4d}'
    msg += ' - ' + f'Patches: {len(patch_infos)}'
    logger.info(msg)
    lock.release()
    return patch_infos


def parse_args(args):
    args = copy.deepcopy(args)
    assert len(args.window_shapes) == len(args.window_steps)
    args.ws, args.hs, args.xs, args.ys = [], [], [], []
    for (w, h), (x, y) in zip(args.window_shapes, args.window_steps):
        for r in args.zooming:
            args.ws.append(round(w * r))
            args.hs.append(round(h * r))
            args.xs.append(round(x * r))
            args.ys.append(round(y * r))

    assert 0 < args.img_iof_thr <= 1
    assert 0 < args.obj_iof_thr <= 1

    if not isinstance(args.dataset_type, list):
        args.dataset_type = [args.dataset_type]
    if not isinstance(args.loading_args, list):
        args.loading_args = [args.loading_args]

    assert not osp.exists(args.dumping_dir), 'Dumping dir ' + \
            f'({args.dumping_dir}) is already existing.'
    args.dumping_img_dir = osp.join(args.dumping_dir, 'images')
    args.dumping_ann_dir = osp.join(args.dumping_dir, 'annotations')
    os.makedirs(args.dumping_img_dir)
    os.makedirs(args.dumping_ann_dir)
    return args


def main():
    parser = argparse.ArgumentParser(description='Image Splitting')
    parser.add_argument('config', type=str, help='The splitting config.')
    args = Config.fromfile(parser.parse_args().config)
    args_ = parse_args(args)

    logger = logging.getLogger('split images')
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = osp.join(args_.dumping_dir, now + '.log')
    lock_path = osp.join(args_.dumping_dir, now + '.lock')
    handlers = [
        logging.StreamHandler(),
        ConcurrentRotatingFileHandler(log_path, mode='w')]
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    contents = load_dataset(args_.dataset_type, args_.loading_args)
    contents_parsed = []
    for largs, cnts in zip(args_.loading_args, contents):
        for cnt in cnts:
            cnt = copy.deepcopy(cnt)
            cnt['img_dir'] = largs['img_dir']
            contents_parsed.append(cnt)

    manager = Manager()
    worker_func = partial(single_process,
                          args=args_,
                          lock=manager.Lock(),
                          prog=manager.Value('i', 0),
                          total=len(contents_parsed),
                          logger=logger)

    logger.info('\n' + args.pretty_text)
    logger.info('Start splitting images!!!')
    start_time = time.time()
    if args_.nproc > 1:
        pool = Pool(args_.nproc)
        patch_infos = pool.map(worker_func, contents_parsed)
        pool.close()
    else:
        patch_infos = list(map(worker_func, contents_parsed))
    patch_infos = reduce(lambda x, y: x+y, patch_infos)
    end_time = time.time()
    logger.info(f'Finish splitting images in {int(end_time - start_time)} second.')
    logger.info(f'Total number of patches: {len(patch_infos)}.')

    args.dump(osp.join(args_.dumping_dir, 'splitting_config.py'))
    with open(osp.join(args_.dumping_ann_dir, 'ori_annfile.pkl'), 'wb') as f:
        pickle.dump(contents, f)
    with open(osp.join(args_.dumping_ann_dir, 'annfile.pkl'), 'wb') as f:
        pickle.dump(patch_infos, f)
    os.remove(lock_path)


if __name__ == '__main__':
    main()
