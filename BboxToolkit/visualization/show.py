import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from .colors import colors_val, random_colors
from .draw import draw_hbb, draw_obb, draw_poly
from ..utils import choice_by_type

EPS = 1e-2


def plt_init(win_name, width, height):
    if win_name is None or win_name == '':
        win_name = str(time.time())
    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    return ax, fig


def get_img_from_fig(fig, width, height):
    stream, _ = fig.canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype=np.uint8)
    img_rgba = buffer.reshape(height, width, 4)
    img, _ = np.split(img_rgba, [3], axis=2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def imshow_bboxes(img,
                  bboxes,
                  labels=None,
                  scores=None,
                  segms=None,
                  class_names=None,
                  colors='green',
                  thickness=1,
                  with_text=True,
                  font_size=10,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    if isinstance(img, np.ndarray):
        img = np.ascontiguousarray(img)
    else:
        img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if isinstance(bboxes, list):
        assert labels is None and scores is None and segms is None
        with_score = True
    else:
        if scores is None:
            with_score = False
        else:
            bboxes = np.concatenate([bboxes, scores[:, None]], axis=1)
            with_score = True

        if labels is None or labels.size == 0:
            bboxes = [bboxes]
            if segms is not None:
                segms = [segms]
        else:
            bboxes = [bboxes[labels == i] for i in range(labels.max()+1)]
            if segms is not None:
                segms = [segms[labels == i] for i in range(labels.max()+1)]

    colors = colors_val(colors)
    if len(colors) == 1:
        colors = colors * len(bboxes)
    assert len(colors) >= len(bboxes)

    draw_func = choice_by_type(
        draw_hbb, draw_obb, draw_poly, bboxes[0], with_score)

    height, width = img.shape[:2]
    ax, fig = plt_init(win_name, width, height)

    if segms is not None:
        for i, cls_segms in enumerate(segms):
            color = np.array(colors[i])
            color = (255 * color).astype(np.uint8)
            for segm in cls_segms:
                mask = segm.astype(bool)
                img[mask] = img[mask] * 0.5 + color * 0.5

    plt.imshow(img)
    for i, cls_bboxes in enumerate(bboxes):
        if with_score:
            cls_bboxes, cls_scores = cls_bboxes[:, :-1], cls_bboxes[:, -1]

        if not with_text:
            texts = None
        else:
            texts = []
            for j in range(len(cls_bboxes)):
                text = f'cls: {i}' if class_names is None else class_names[i]
                if with_score:
                    text += f'|{cls_scores[j]:.02f}'
                texts.append(text)

        draw_func(ax, cls_bboxes, texts, colors[i], thickness, font_size)

    drawed_img = get_img_from_fig(fig, width, height)
    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, drawed_img)

    plt.close(fig)
    return drawed_img
