import cv2
import numpy as np
from collections.abc import Iterable

from .base import BaseBbox, AlignPOLY

pi = np.pi


class OBB(BaseBbox):
    Bdim = 5

    def __init__(self, bboxes, scores=None):
        bboxes = self.regulate_obb(bboxes)
        super(OBB, self).__init__(bboxes, scores)

    @staticmethod
    def regulate_obb(obboxes):
        ctr, w, h, t = np.split(obboxes, (2, 3, 4), axis=-1)
        w_reg = np.where(w > h, w, h)
        h_reg = np.where(w > h, h, w)
        t_reg = np.where(w > h, t, t+pi/2)

        t_reg = t_reg + pi/2
        t_reg = t_reg % pi
        t_reg = t_reg - pi/2

        return np.concatenate([ctr, w_reg, h_reg, t_reg], axis=-1)

    @classmethod
    def gen_empty(cls, with_scores=False):
        bboxes = np.zeros((0, 5), dtype=np.float32)
        scores = (None if not with_scores else
                  np.zeros((0, ), dtype=np.float32))
        return cls(bboxes, scores)

    @classmethod
    def gen_random(cls, shape, scale=1, with_scores=False):
        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape, )
        shape = shape + (5, )

        obboxes = np.random.random(shape).astype(np.float32)
        obboxes *= np.array(
            [scale, scale, scale, scale, pi], dtype=np.float32)
        scores = (None if not with_scores else
                  np.random.random(shape[:-1]).astype(np.float32))

        return cls(obboxes, scores)

    @staticmethod
    def bbox_from_poly(polys):
        order = polys.shape[:-1]
        num_points = polys.shape[-1] // 2
        polys = polys.reshape(-1, num_points, 2)
        polys = polys.astype(np.float32)

        obboxes = []
        for p in polys:
            (x, y), (w, h), angle = cv2.minAreaRect(p)
            obboxes.append([x, y, w, h, angle/180*pi])
        obboxes = np.array(obboxes, dtype=np.float32).reshape(*order, 5)

        ctr, w, h, t = np.split(obboxes, (2, 3, 4), axis=-1)
        w_reg = np.where(w > h, w, h)
        h_reg= np.where(w > h, h, w)
        t_reg= np.where(w > h, t, t+pi/2)
        return np.concatenate([ctr, w_reg, h_reg, t_reg], axis=-1)

    @staticmethod
    def bbox_to_poly(obboxes):
        ctr, w, h, t = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(t), np.sin(t)

        vec1 = np.concatenate(
            [w/2 * Cos, w/2 * Sin], axis=-1)
        vec2 = np.concatenate(
            [-h/2 * Sin, h/2 * Cos], axis=-1)

        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return np.concatenate([pt1, pt2, pt3, pt4], axis=-1)

    def areas(self):
        bboxes = self.bboxes
        return bboxes[..., 2] * bboxes[..., 3]

    # self-changing functions
    def flip(self, W, H, direction='horizontal'):
        assert direction in ['horizontal', 'vertical', 'diagonal']
        bboxes = self.bboxes

        flipped = bboxes.copy()
        if direction == 'horizontal':
            flipped[..., 0] = W - bboxes[..., 0]
            flipped[..., 4] = -flipped[..., 4]
        elif direction == 'vertical':
            flipped[..., 1] = H - bboxes[..., 1]
            flipped[..., 4] = -flipped[..., 4]
        else:
            flipped[..., 0] = W - bboxes[..., 0]
            flipped[..., 1] = H - bboxes[..., 1]

        return type(self)(self.regulate_obb(flipped), self.scores)

    def translate(self, x, y):
        new_bboxes = self.bboxes.copy()
        new_bboxes[..., :2] += np.array((x, y), dtype=np.float32)
        return type(self)(new_bboxes, self.scores)

    def rotate(self, center, angle):
        M = cv2.getRotationMatrix2D(center, angle, 1)
        ctr, w, h, t = np.split(self.bboxes, (2, 3, 4), axis=-1)
        ctr = self.warp_pts(ctr, M)
        t = t - angle / 180 * pi
        bboxes = np.concatenate([ctr, w, h, t], aixs=-1)
        return type(self)(self.regulate_obb(bboxes), self.scores)

    def rescale(self, scales):
        new_bboxes = self.bboxes.copy()
        if isinstance(scales, (tuple, list)):
            assert len(scales) == 2
            new_bboxes[..., :4] *= np.array(scales*2, dtype=np.float32)
        else:
            new_bboxes[..., :4] *= scales
        return type(self)(new_bboxes, self.scores)
