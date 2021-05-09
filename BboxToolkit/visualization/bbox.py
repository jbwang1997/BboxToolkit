# Modified from mmcv.visualization
# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np

from .base import color_val
from .base import imshow
from ..utils import get_bbox_type
from ..transforms import bbox2type


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  max_size=1000,
                  out_file=None):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, n).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    if isinstance(img, np.ndarray):
        img = np.ascontiguousarray(img)
    else:
        img = cv2.imread(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])

        _bboxes = _bboxes[:_top_k]
        _polys = bbox2type(_bboxes, 'poly').astype(np.int32)
        _polys = _polys.reshape(-1, _polys.shape[1] // 2, 2)
        img = cv2.polylines(img, _polys, isClosed=True,
                            color=colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time, max_size)
    if out_file is not None:
        cv2.imwrite(out_file, img)
    return img


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      scores=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      max_size=1000,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (without scores)
        labels (ndarray): Labels of bboxes.
        scores (ndarray): Scores of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert get_bbox_type(bboxes) != 'notype'
    if isinstance(img, np.ndarray):
        img = np.ascontiguousarray(img)
    else:
        img = cv2.imread(img)

    if score_thr > 0 and scores is not None:
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        scores = scores[inds]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    polys = bbox2type(bboxes, 'poly')
    for i, (poly, label) in enumerate(zip(polys, labels)):
        poly_int = poly.reshape(1, poly.size//2, 2).astype(np.int32)
        img = cv2.polylines(img, poly_int, isClosed=True,
                            color=bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if scores is not None:
            label_text += f'|{scores[i]:.02f}'

        highest_pos_ind = np.argmax(poly_int[0, :, 1])
        highest_pos = poly_int[0, highest_pos_ind]
        cv2.putText(img, label_text, (highest_pos[0], highest_pos[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        imshow(img, win_name, wait_time, max_size)
    if out_file is not None:
        cv2.imwrite(out_file, img)
    return img
