import numpy as np
from collections.abc import Iterable

from .base import BaseBbox


class HBB(BaseBbox):
    Bdim = 4

    @classmethod
    def gen_empty(cls, with_scores=False):
        bboxes = np.zeros((0, 4), dtype=np.float32)
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
        xmin = np.minimum(border[..., 0], border[..., 2])
        ymin = np.minimum(border[..., 1], border[..., 3])
        xmax = np.maximum(border[..., 0], border[..., 2])
        ymax = np.maximum(border[..., 1], border[..., 3])
        scores = (None if not with_scores else
                  np.random.random(shape[:-1]).astype(np.float32))

        return cls(np.stack([xmin, ymin, xmax, ymax], axis=1), scores)

    @staticmethod
    def bbox_from_poly(polys):
        shape = polys.shape
        polys = polys.reshape(*shape[:-1], shape[-1]//2, 2)
        lt_point = np.min(polys, axis=-2)
        rb_point = np.max(polys, axis=-2)
        return np.concatenate([lt_point, rb_point], axis=-1)

    @staticmethod
    def bbox_to_poly(bboxes):
        l, t, r, b = np.split(bboxes, 4, axis=-1)
        return np.stack([l, t, r, t, r, b, l, b], axis=-1)

    def areas(self):
        bboxes = self.bboxes
        return (bboxes[..., 2] - bboxes[..., 0]) * \
                (bboxes[..., 3] - bboxes[..., 1])

    def flip_(self, W, H, direction='horizontal'):
        assert direction in ['horizontal', 'vertical', 'diagonal']
        bboxes = self.bboxes

        flipped = bboxes.copy()
        if direction == 'horizontal':
            flipped[..., 0::4] = W - bboxes[..., 2::4]
            flipped[..., 2::4] = W - bboxes[..., 0::4]
        elif direction == 'vertical':
            flipped[..., 1::4] = H - bboxes[..., 3::4]
            flipped[..., 3::4] = H - bboxes[..., 1::4]
        else:
            flipped[..., 0::4] = W - bboxes[..., 2::4]
            flipped[..., 1::4] = H - bboxes[..., 3::4]
            flipped[..., 2::4] = W - bboxes[..., 0::4]
            flipped[..., 3::4] = H - bboxes[..., 1::4]

        self.bboxes = flipped

    def translate_(self, x, y):
        self.bboxes += np.array((x, y) * 2, dtype=np.float32)

    def rescale_(self, scales):
        if isinstance(scales, (tuple, list)):
            assert len(scales) == 2
            self.bboxes *= np.array(scales * 2, dtype=np.float32)
        else:
            self.bboxes *= scales
