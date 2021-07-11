'''Visualize Bboxes in images.

Reference:
    mmcv: https://github.com/open-mmlab/mmcv
    mmdetection: https://github.com/open-mmlab/mmdetection
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from .colors import colors_val
from ..structures import BaseBbox

EPS = 1e-2


def imshow_with_bboxes(img,
                       bboxes,
                       labels=None,
                       scores=None,
                       score_thr=0,
                       class_names=None,
                       colors='green',
                       thickness=1,
                       font_size=10,
                       without_text=False,
                       win_name='',
                       show=True,
                       wait_time=0,
                       out_file=None):
    assert isinstance(bboxes, BaseBbox), \
            f'bboxes should be a instance of BaseBbox.'
    num = len(bboxes)

    if labels is not None:
        assert labels.shape[0] == num
    if scores is not None:
        assert scores.shape[0] == num

    assert 0 <= score_thr <= 1, f'score_thr({score_thr}) is invalid.'
    if score_thr > 0:
        bboxes = bboxes[scores >= score_thr]
        labels = None if labels is None else \
                labels[scores >= score_thr]
        scores = None if scores is None else \
                scores[scores >= score_thr]

    colors = colors_val(colors)

    if isinstance(img, str):
        img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img)
    width, height = img.shape[1], img.shape[0]

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    plt.imshow(img)

    bbox_colors = []
    bbox_texts = []
    for i in range(num):
        label = 0 if labels is None else labels[i]
        bbox_colors.append(
            colors[label] if len(colors) > 1 else colors[0])

        if without_text:
            bbox_texts.append(None)
            continue

        text = ''
        text += class_names[label] if class_names is not None \
                else f'class {label}'
        if scores is not None:
            text += f'|{scores[i]}'
        bbox_texts.append(text)
    bboxes.visualize(ax, bbox_texts, bbox_colors, thickness=thickness,
                     font_size=font_size)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, _ = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)
