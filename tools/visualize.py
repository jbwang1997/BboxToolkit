import BboxToolkit as bt
import os
import os.path as osp
import argparse
import numpy as np

from random import shuffle
from multiprocessing import Pool, Manager
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description='visualization')

    # arguments for loading data
    parser.add_argument('--load_type', type=str, help='dataset and save form')
    parser.add_argument('--img_dir', type=str, help='path to images')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotations')
    parser.add_argument('--classes', type=str, default=None,
                        help='the classes to load, a filepath or class names joined by `|`')
    parser.add_argument('--nproc', type=int, default=10,
                        help='the procession number for loading data')

    # arguments for selecting content
    parser.add_argument('--skip_empty', action='store_true',
                        help='whether show images without objects')
    parser.add_argument('--random_vis', action='store_true',
                        help='whether to shuffle the order of images')
    parser.add_argument('--ids', type=str, default=None,
                        help='choice id to visualize')
    parser.add_argument('--show_off', action='store_true',
                        help='stop showing images')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='whether to save images and where to save images')
    parser.add_argument('--vis_nproc', type=int, default=10,
                        help='the procession number for visualizing')

    # arguments for visualisation
    parser.add_argument('--class_names', type=str, default=None,
                        help='class names shown in picture')
    parser.add_argument('--score_thr', type=float, default=0.2,
                        help='the score threshold for bboxes')
    parser.add_argument('--colors', type=str, default='green',
                        help='the thickness for bboxes')
    parser.add_argument('--thickness', type=float, default=1.,
                        help='the thickness for bboxes')
    parser.add_argument('--text_off', action='store_true',
                        help='without text visualization')
    parser.add_argument('--font_size', type=float, default=10,
                        help='the thickness for font')
    parser.add_argument('--wait_time', type=int, default=0,
                        help='wait time for showing images')
    args = parser.parse_args()
    assert args.load_type is not None, "argument load_type can't be None"
    assert args.img_dir is not None, "argument img_dir can't be None"
    assert args.save_dir or (not args.show_off)

    return args


def single_vis(content, ids, img_dir, save_dir, class_names, score_thr, colors,
               thickness, text_off, font_size, show_off, wait_time, lock, prog, total):
    if ids is not None and content['id'] not in ids:
        pass
    else:
        imgpath = osp.join(img_dir, content['filename'])
        out_file = osp.join(save_dir, content['filename']) \
                if save_dir else None
        if 'ann' in content:
            ann = content['ann']
            bboxes = ann['bboxes']
            labels = ann['labels']
            scores = ann.get('scores', None)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float)
            labels = np.zeros((0, ), dtype=np.int)
            scores = None

        bt.imshow_bboxes(imgpath, bboxes, labels, scores,
                         class_names=class_names,
                         score_thr=score_thr,
                         colors=colors,
                         thickness=thickness,
                         with_text=(not text_off),
                         font_size=font_size,
                         show=(not show_off),
                         wait_time=wait_time,
                         out_file=out_file)

    lock.acquire()
    prog.value += 1
    msg = f'({prog.value/total:3.1%} {prog.value}:{total})'
    msg += ' - '  + f"Filename: {content['filename']}"
    print(msg)
    lock.release()


def main():
    args = parse_args()

    print(f'{args.load_type} loading!')
    load_func = getattr(bt.datasets, 'load_'+args.load_type)
    contents, classes = load_func(
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        classes=args.classes,
        nproc=args.nproc)

    class_names = classes if args.class_names is None \
            else bt.get_classes(args.class_names)
    assert len(class_names) == len(classes)

    if args.ids is None:
        ids = None
    elif osp.isfile(args.ids):
        with open(args.ids, 'r') as f:
            ids = [l.strip() for l in f]
    else:
        ids = args.ids.split('|')

    if args.skip_empty:
        contents = [content for content in contents
                    if content['ann']['bboxes'].size > 0]
    if args.random_vis:
        shuffle(contents)
    if args.save_dir and (not osp.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    manager = Manager()
    _vis_func = partial(single_vis,
                        ids=ids,
                        img_dir=args.img_dir,
                        save_dir=args.save_dir,
                        class_names=class_names,
                        score_thr=args.score_thr,
                        colors=args.colors,
                        thickness=args.thickness,
                        text_off=args.text_off,
                        font_size=args.font_size,
                        show_off=args.show_off,
                        wait_time=args.wait_time,
                        lock=manager.Lock(),
                        prog=manager.Value('i', 0),
                        total=len(contents))
    if args.show_off and args.vis_nproc > 1:
        pool = Pool(args.vis_nproc)
        pool.map(_vis_func, contents)
        pool.close()
    else:
        list(map(_vis_func, contents))


if __name__ == '__main__':
    main()
