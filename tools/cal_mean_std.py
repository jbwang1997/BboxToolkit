import os
import cv2
import pickle
import argparse
import numpy as np
import os.path as osp

from multiprocessing import Pool
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description='get imageset norm cfg')
    parser.add_argument('--img_dir', type=str, help='path to images')
    parser.add_argument('--to_rgb', type=int, default=1,
                        help='convert bgr images to rgb')
    parser.add_argument('--long_edge_size', type=int, default=None,
                        help='force the long edge to regular size')
    parser.add_argument('--short_edge_size', type=int, default=None,
                        help='force the short edge to regular size')
    parser.add_argument('--nproc', type=int, default=10,
                        help='the processing number')
    parser.add_argument('--out', type=str, default=None,
                        help='save calculation results in pkl')
    args = parser.parse_args()

    assert args.img_dir is not None, "argument img_dir can't be None"
    assert args.out is None or args.out.endswith('.pkl'), \
            'argument out must be a pkl file'
    return args


def _rescale_size(w, h, long_edge_size, short_edge_size):
    if (long_edge_size is not None) and (short_edge_size is not None):
        assert long_edge_size >= short_edge_size

    scale_factor = []
    if long_edge_size is not None:
        scale_factor.append(long_edge_size / max(w, h))
    if short_edge_size is not None:
        scale_factor.append(short_edge_size / min(w, h))
    scale_factor = min(scale_factor) if scale_factor else 1

    new_w = int(w * float(scale_factor) + 0.5)
    new_h = int(h * float(scale_factor) + 0.5)
    return new_w, new_h


def _single_func(imgpath, to_rgb, long_edge_size, short_edge_size):
    print(imgpath)
    img = cv2.imread(imgpath)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (long_edge_size is not None) or (short_edge_size is not None):
        h, w, _ = img.shape
        new_size = _rescale_size(w, h, long_edge_size, short_edge_size)
        img = cv2.resize(img, new_size)
    return cv2.meanStdDev(img)


def main():
    args = parse_args()

    imgs = []
    for f in os.listdir(args.img_dir):
        if osp.splitext(f)[-1] in ['.png', '.jpg', '.tif']:
            imgs.append(osp.join(args.img_dir, f))
    func = partial(_single_func,
                   to_rgb=args.to_rgb>0,
                   long_edge_size=args.long_edge_size,
                   short_edge_size=args.short_edge_size)

    if args.nproc > 1:
        pool = Pool(args.nproc)
        results = pool.map(func, imgs)
    else:
        results = list(map(func, imgs))

    means, stds = list(zip(*results))
    means = np.concatenate(means, axis=-1)
    stds = np.concatenate(stds, axis=-1)
    total_mean = means.mean(-1)
    total_std = np.sqrt((stds * stds).mean(-1))
    print('images mean: ', total_mean.tolist())
    print('images std: ', total_std.tolist())

    if args.out is not None:
        save_dict = dict(
            mean=total_mean.tolist(),
            std=total_std.tolist())
        pickle.dump(save_dict, args.out)


if __name__ == '__main__':
    main()
