import cv2
import numpy as np
from collections.abc import Iterable

from .base import AlignPOLY


class P4POLY(AlignPOLY):
    Bdim = 8

    def __init__(self, bboxes, scores=None):
        super(AlignPOLY, self).__init__(bboxes, scores)

    @classmethod
    def gen_empty(cls, with_scores=False):
        bboxes = np.zeros((0, 8), dtype=np.float32)
        scores = (None if not with_scores else
                  np.zeros((0, ), dtype=np.float32))
        return cls(bboxes, scores)

    @classmethod
    def gen_random(cls, shape, scale=1, with_scores=False):
        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape, )
        shape = shape + (4, )

        border = scale * np.random.random(shape).astype(np.float32)
        factor = np.random.random(shape).astype(np.float32)

        xmin = np.minimum(border[..., 0], border[..., 2])
        ymin = np.minimum(border[..., 1], border[..., 3])
        xmax = np.maximum(border[..., 0], border[..., 2])
        ymax = np.maximum(border[..., 1], border[..., 3])

        w = xmax - xmin
        h = ymax - ymin

        p1_x = xmin + factor[..., 0] * w
        p2_y = ymin + factor[..., 1] * h
        p3_x = xmax - factor[..., 2] * w
        p4_y = ymax - factor[..., 3] * h

        polys = np.stack([p1_x, ymin, xmax, p2_y,
                          p3_x, ymax, xmin, p4_y], axis=1)
        scores = (None if not with_scores else
                  np.random.random(shape[:-1]).astype(np.float32))
        return cls(polys, scores)

    @staticmethod
    def bbox_from_poly(polys):
        if polys.shape[-1] == 8:
            return polys.copy()

        order = polys.shape[:-1]
        num_points = polys.shape[-1] // 2
        polys = polys.reshape(-1, num_points, 2)

        p4_polys = []
        for p in polys:
            rect = cv2.minAreaRect(p)
            p4_polys.append(cv2.boxPoints(rect))

        return np.array(p4_polys, dtype=np.float32).reshape(*order, 8)
