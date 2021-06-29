import cv2
import numpy as np

from .base import BaseBbox
from .poly import POLY


class HBB(BaseBbox):
    '''
    Horizontal Bounding Box (HBB): Present horizontal boudning boxes
    by their left top points and right bottom points.

    Args:
        bboxes (ndarray (n, 4)): contain the left top and right bottom
            point coordinates.
    '''

    def __init__(self, bboxes):
        assert isinstance(bboxes, np.ndarray)
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4

        # Copy and convert ndarray type of float32.
        bboxes = bboxes.astype(np.float32)
        self.bboxes = bboxes

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'len={len(self)}, '
        s += f'bboxes={self.bboxes})'
        return s

    def __getitem__(self, index):
        '''see :func:`BaseBbox.__getitem__`'''
        return HBB(self.bboxes[index])

    def __len__(self):
        '''Number of Bboxes.'''
        return self.bboxes.shape[0]

    def to_poly(self):
        '''Output the Bboxes polygons (list[list[np.ndarry]]).'''
        l, t, r, b = np.split(self.bboxes, 4, axis=-1)
        pts = np.stack([l, t, r, t, r, b, l, b], axis=-1)
        return [[p] for p in pts]

    @classmethod
    def from_poly(cls, polys):
        '''Create a Bbox instance from polygons (list[list[np.ndarray]]).'''
        if not polys:
            return cls.gen_empty()

        hbbs = []
        for poly in polys:
            pts = np.concatenate(poly).reshape(-1, 2)
            lt_points = pts.min(axis=0)
            rb_points = pts.max(axis=0)
            hbbs.append(np.concatenate([lt_points, rb_points]))
        return HBB(np.stack(hbbs, axis=0))

    def copy(self):
        '''Copy this instance.'''
        return HBB(self.bboxes)

    @classmethod
    def gen_empty(cls):
        '''Create a Bbox instance with len == 0.'''
        bboxes = np.zeros((0, 4), dtype=np.float32)
        return HBB(bboxes)

    def areas(self):
        '''Return areas of Bboxes.'''
        bboxes = self.bboxes
        return (bboxes[..., 2] - bboxes[..., 0]) * \
                (bboxes[..., 3] - bboxes[..., 1])

    def rotate(self, x, y, angle, keep_btype=True):
        '''see :func:`BaseBbox.rotate`'''
        # List the points of HBBs.
        l, t, r, b = np.split(self.bboxes, 4, axis=-1)
        pts = np.stack([l, t, r, t, r, b, l, b], axis=-1)
        pts = pts.reshape(-1, 4, 2)

        # Get the roatation matrix and rotate all points
        M = cv2.getRotationMatrix2D((x, y), angle, 1)
        rotated_pts = cv2.transform(pts, M)

        # Convert points to Output.
        if keep_btype:
            lt_points = rotated_pts.min(axis=1)
            rb_points = rotated_pts.max(axis=1)
            return HBB(np.concatenate([lt_points, rb_points], axis=1))
        else:
            rotated_pts = rotated_pts.reshape(-1, 8)
            return POLY([[p] for p in rotated_pts])

    def warp(self, M, keep_btype=False):
        '''see :func:`BaseBbox.warp`'''
        # List the points of HBBs.
        l, t, r, b = np.split(self.bboxes, 4, axis=-1)
        pts = np.stack([l, t, r, t, r, b, l, b], axis=-1)
        pts = pts.reshape(-1, 4, 2)

        # Warp points.
        if M.shape[0] == 2:
            warped_pts = cv2.transform(pts, M)
        elif M.shape[0] == 3:
            warped_pts = cv2.prospectiveTransform(pts, M)
        else:
            raise ValueError(f'Wrong M shape {M.shape}')

        # Transform points to Output.
        if keep_btype:
            lt_points = warped_pts.min(axis=1)
            rb_points = warped_pts.max(axis=1)
            return HBB(np.concatenate([lt_points, rb_points], axis=1))
        else:
            warped_pts = warped_pts.reshape(-1, 8)
            return POLY([[p] for p in warped_pts])

    def flip(self, W, H, direction='horizontal'):
        '''see :func:`BaseBbox.flip`'''
        assert direction in ['horizontal', 'vertical', 'diagonal']
        bboxes = self.bboxes

        flipped = bboxes.copy()
        if direction == 'horizontal':
            flipped[:, 0] = W - bboxes[:, 2]
            flipped[:, 2] = W - bboxes[:, 0]
        elif direction == 'vertical':
            flipped[:, 1] = H - bboxes[:, 3]
            flipped[:, 3] = H - bboxes[:, 1]
        else:
            flipped[:, 0] = W - bboxes[:, 2]
            flipped[:, 1] = H - bboxes[:, 3]
            flipped[:, 2] = W - bboxes[:, 0]
            flipped[:, 3] = H - bboxes[:, 1]

        return HBB(flipped)

    def translate(self, x, y):
        '''see :func:`BaseBbox.translate`'''
        bboxes = self.bboxes + np.asarray([x, y, x, y], dtype=np.float32)
        return HBB(bboxes)

    def resize(self, ratios):
        '''see :func:`BaseBbox.resize`'''
        if isinstance(ratios, (tuple, list)):
            assert len(ratios) == 2
            bboxes = self.bboxes * np.asarray(ratios * 2, dtype=np.float32)
        else:
            bboxes = self.bboxes * ratios
        return HBB(bboxes)
