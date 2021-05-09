import numpy as np
from collections.abc import Iterable

from .base import AlignPOLY


class PT(AlignPOLY):
    Bdim = 2

    def __init__(self, bboxes, scores=None):
        super(AlignPOLY, self).__init__(bboxes, scores)

    @classmethod
    def gen_empty(cls, with_scores=False):
        bboxes = np.zeros((0, 2), dtype=np.float32)
        scores = (None if not with_scores else
                  np.zeros((0, ), dtype=np.float32))
        return cls(bboxes, scores)

    @classmethod
    def gen_random(cls, shape, scale=1, with_scores=False):
        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape, )
        shape = shape + (2, )

        pts = scale * np.random.random(shape).astype(np.float32)
        scores = (None if not with_scores else
                  np.random.random(shape[:-1]).astype(np.float32))
        return cls(pts, scores)

    @staticmethod
    def bbox_from_poly(polys):
        x = polys[..., 0::2].mean(axis=-1)
        y = polys[..., 1::2].mean(axis=-1)
        return np.stack([x,y], axis=-1)

    def areas(self):
        shape = self.bboxes.shape
        return np.zeros(shape[:-1], dtype=np.float32)
