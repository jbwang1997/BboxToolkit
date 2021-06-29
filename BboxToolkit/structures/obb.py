import cv2
import numpy as np

from .base import BaseBbox
from .poly import POLY
from ..utils.configs import pi


class OBB(BaseBbox):
    '''
    Oriented Bouding Box (OBB): Implement OBB operations and store OBBs
    in form of X, Y, W, H, T where X, Y are the center coordinates, W, H
    are the length of two sides, T is the theta between W side and X axis.

    Args:
        bboxes (ndarray (n, 5)): oriented bboxes in form (X, Y, W, H, T).
    '''

    def __init__(self, bboxes):
        assert isinstance(bboxes, np.ndarray)
        assert bboxes.ndim == 2 and bboxes.shape[1] == 5

        # Copy and standardize obboxes.
        bboxes = bboxes.astype(np.float32)
        if bboxes.size > 0:
            bboxes = self._standardize(bboxes)
        self.bboxes = bboxes

    @staticmethod
    def _standardize(obbs):
        '''Standardize the OBBs, which W is the length of longer side, H
           is the length of shorter side, T is the clockwise angle between
           x axis and longer size. the Theta is in interval [-pi/2, pi/2).

        Args:
            obbs (np.ndarray (-1, 5)): Oriented boxes.

        Returns:
            Standardized OBBs
        '''
        ctr, w, h, t = np.split(obbs, (2, 3, 4), axis=-1)
        reg_w = np.where(w > h, w, h)
        reg_h = np.where(w > h, h, w)
        reg_t = np.where(w > h, t, t+pi/2)
        reg_t = (reg_t + pi/2) % pi - pi/2
        return np.concatenate([ctr, reg_w, reg_h, reg_t], axis=-1)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'len={len(self)}, '
        s += f'bboxes={self.bboxes})'
        return s

    def __getitem__(self, index):
        '''see :func:`BaseBbox.__getitem__`'''
        return OBB(self.bboxes[index])

    def __len__(self):
        '''Number of Bboxes.'''
        return self.bboxes.shape[0]

    def to_poly(self):
        '''Output the Bboxes polygons (list[list[np.ndarry]]).'''
        polys = []
        for bbox in self.bboxes:
            x, y, w, h, theta = bbox
            pts = cv2.boxPoints((x, y), (w, h), theta/pi*180)
            polys.append([pts.reshape(-1)])
        return polys

    @classmethod
    def from_poly(cls, polys):
        '''Create a Bbox instance from polygons (list[list[np.ndarray]]).'''
        if not polys:
            return cls.gen_empty()

        obbs = []
        for poly in polys:
            pts = np.concatenate(poly).reshape(-1, 2)
            (x, y), (w, h), angle = cv2.minAreaRect(pts)
            obbs.append([x, y, w, h, angle/180*pi])
        return OBB(np.asarray(obbs))

    def copy(self):
        '''Copy this instance.'''
        obbs = OBB.gen_empty()
        obbs.bboxes = self.bboxes.copy()
        return obbs

    @classmethod
    def gen_empty(cls):
        '''Create a Bbox instance with len == 0.'''
        bboxes = np.zeros((0, 5), dtype=np.float32)
        return OBB(bboxes)

    def areas(self):
        '''ndarry: areas of each instance.'''
        bboxes = self.bboxes
        return bboxes[:, 2] * bboxes[:, 3]

    def rotate(self, x, y, angle, keep_btype=True):
        '''see :func:`BaseBbox.rotate`'''
        M = cv2.getRotationMatrix2D((x, y), angle, 1)
        ctr, w, h, t = np.split(self.bboxes, (2, 3, 4), axis=1)

        # Rotate OBBs
        ctr = ctr[:, None, :]
        rotated_ctr = cv2.transform(ctr, M)[:, 0, :]
        rotated_t = t - angle / 180 * pi
        rotated_bboxes = np.concatenate(
            [rotated_ctr, w, h, rotated_t], axis=1)

        output = OBB(rotated_bboxes)
        if not keep_btype:
            output = output.to_type(POLY)
        return output

    def warp(self, M, keep_btype=False):
        '''see :func:`BaseBbox.warp`'''
        ctr, w, h, t = np.split(self.bboxes, (2, 3, 4), axis=1)
        Cos, Sin = np.cos(t), np.sin(t)
        # vectors to W, H direction.
        vec1 = np.concatenate([w/2 * Cos, w/2 * Sin], axis=1)
        vec2 = np.concatenate([-h/2 * Sin, h/2 * Cos], axis=1)

        # List the points of OBBs.
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        pts = np.stack([pt1, pt2, pt3, pt4], axis=1)

        # Warp points.
        if M.shape[0] == 2:
            warped_pts = cv2.transform(pts, M)
        elif M.shape[0] == 3:
            warped_pts = cv2.prospectTransform(pts, M)

        # Convert to ouputs.
        polys = warped_pts.reshape(-1, 8)
        polys = [[p] for p in polys]
        if keep_btype:
            return OBB.from_poly(polys)
        else:
            return POLY(polys)

    def flip(self, W, H, direction='horizontal'):
        '''see :func:`BaseBbox.flip`'''
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

        return OBB(flipped)

    def translate(self, x, y):
        '''see :func:`BaseBbox.translate`'''
        bboxes = self.bboxes.copy()
        bboxes[..., :2] += np.asarray((x, y), dtype=np.float32)
        return OBB(bboxes)

    def resize(self, ratios):
        '''see :func:`BaseBbox.resize`'''
        bboxes = self.bboxes.copy()
        if isinstance(ratios, (tuple, list)):
            assert len(ratios) == 2
            bboxes[..., :4] *= np.asarray(ratios*2, dtype=np.float32)
        else:
            bboxes[..., :4] *= ratios
        return OBB(bboxes)
