import os
import cv2
import sys
import copy
import logging
import argparse
import numpy as np
import os.path as osp

from math import ceil
from itertools import product, count

from BboxToolkit.utils import Config
from BboxToolkit.ops import bbox_overlaps
from BboxToolkit.structures import BaseBbox, HBB
from BboxToolkit.datasets import load_dataset, dump_dataset


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
    if not np.any(win_iofs > iof_thr):
        win_iofs[abs(win_iofs - win_iofs.max()) < 0.01] = 1
    return windows[win_iofs > iof_thr]


def crop_objects(content, windows, iof_thr, skip_empty, img_ext):
    ann = content.pop('ann')
    bboxes = ann.get('bboxes', HBB.gen_empty())
    iofs = bbox_overlaps(bboxes, windows, mode='iof')

    counter = count()
    win_contents = []
    for i in range(len(windows)):
        win_iofs = iofs[:, i]
        retain_idx = np.nonzero(win_iofs >= iof_thr)[0]
        if len(retain_idx) == 0 and skip_empty:
            continue

        win_content = copy.deepcopy(content)
        ori_filename = win_content['filename']
        img_id, _ = osp.splitext(ori_filename)
        win_content['ori_filename'] = ori_filename
        win_content['fileanme'] = img_id + f'_{next(counter):04d}' + img_ext

        x_start, y_start, x_stop, y_stop = windows.bboxes[i]
        win_content['x_start'] = x_start
        win_content['y_start'] = y_start
        win_content['width'] = x_stop - x_start
        win_content['height'] = y_stop - y_start

        win_ann = dict()
        for k, v in ann.items():
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


def crop_and_save_img(img_dir, content, win_contents, padding_value, save_dir):
    if len(win_contents) == 0:
        return

    img = cv2.imread(osp.join(img_dir, content['filename']))
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


# def single_process(content, img_prog, logger, args):
    # W, H = content['width'], content['height']
    # windows = slide_windows(W, H)
    # win_contents = crop_objects(windows, args.
    # img_prog += 1


def assert_args(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Splitting')
    parser.add_argument('config', type=str, help='The splitting config.')
    args = Config.from_file(parser.parse_args().config)
    assert_args(args)

    # contents = load_dataset(args.dataset_type, args.

