import cv2
import numpy as np
from .base import BaseBbox


class HBB(BaseBbox):

    def __init__(self, bboxes):
        assert isinstance(bboxes, np.ndarray)
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4

        # Copy and convert np.ndarray type of float32.
        bboxes = bboxes.astype(np.float32)
        self.bboxes = bboxes

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'bboxes={self.bboxes})'
        return s

    def __getitem__(self, index):
        '''see :func:`BaseBbox.__getitem__`'''
        bboxes = self.bboxes[index]
        return HBB(bboxes)

    def __len__(self):
        '''Number of Bboxes.'''
        return self.bboxes.shape[0]

    def to_poly(self):
        '''Output the Bboxes polygons (list[list[np.ndarry]]).'''
        polys = []
        for bbox in self.bboxes:
            xmin, ymin, xmax, ymax = bbox
            polys.append([np.array([
                xmin, ymin,
                xmax, ymin,
                xmax, ymax,
                xmin, ymax], dtype=np.float32)])
        return polys


    @classmethod
    def from_poly(cls, polys):
        '''Create a Bbox instance from polygons (list[list[np.ndarray]]).'''
        hbbs = []
        for poly in polys:
            pts = np.concatenate(poly).reshape(-1, 2)
            lt_points = pts.min(axis=0)
            rb_points = pts.max(axis=0)
            hbbs.append(np.concatenate([lt_points, rb_points]))

        hbbs = np.stack(hbbs, axis=0)
        return HBB(hbbs)

    def copy(self):
        '''Copy this instance.'''
        return HBB(self.bboxes)

    def gen_empty(self):
        '''Create a Bbox instance with len == 0.'''
        bboxes = np.zeros((0, 4), dtype=np.float32)
        return HBB(bboxes)

    def areas(self):
        '''Return areas of Bboxes.'''
        bboxes = self.bboxes
        return (bboxes[..., 2] - bboxes[..., 0]) * \
                (bboxes[..., 3] - bboxes[..., 1])

    def warp(self, M):
        '''see :func:`BaseBox.warp`'''
        # List the points of HBBs.
        l, t, r, b = np.split(self.bboxes, 4, axis=-1)
        pts = np.stack([l, t, r, t, r, b, l, b], axis=-1)
        pts = pts.reshape(-1, 4, 2)

        # Warp points
        if M.shape[0] == 2:
            warped_pts = cv2.transform(pts, M)
        elif M.shape[0] == 3:
            warped_pts = cv2.prospectivetransform(pts, M)
        else:
            raise ValueError(f'Wrong M shape of {M.shape}')

        # Transform points to HBB
        lt_points = warped_pts.min(axis=1)
        rb_points = warped_pts.max(axis=1)
        return HBB(np.concatenate([lt_points, rb_points], axis=-1))

    def flip(self, W, H, direction='horizontal'):
        '''see :func:`BaseBbox.flip`'''
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

        return HBB(flipped)

    def translate(self, x, y):
        '''see :func:`BaseBbox.translate`'''
        bboxes = self.bboxes + np.array([x, y, x, y], dtype=np.float32)
        return HBB(bboxes)

    def resize(self, ratios):
        '''see :func:`BaseBbox.resize`'''
        if isinstance(ratios, (tuple, list)):
            assert len(ratios) == 2
            bboxes = self.bboxes * np.array(ratios * 2, dtype=np.float32)
        else:
            bboxes = self.bboxes * ratios
        return HBB(bboxes)
