import cv2
import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from .base import BaseBbox
from .poly import POLY


@BaseBbox.register_bbox_cls()
class PT(BaseBbox):
    '''
    Points (PT): Treat targets as points. Restore the coordinates of points.

    Args:
        bboxes (ndarray (n, 2)): contain the coordinates of points.
    '''

    def __init__(self, bboxes):
        assert isinstance(bboxes, np.ndarray)
        assert bboxes.ndim == 2 and bboxes.shape[1] == 2

        # Copy and convert ndarray type to float32
        bboxes = bboxes.astype(np.float32)
        self.bboxes = bboxes

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'len={len(self)}, '
        s += f'bboxes={self.bboxes})'
        return s

    def __getitem__(self, index):
        '''see :func:`BaseBbox.__getitem__`'''
        return PT(self.bboxes[index])

    def __len__(self):
        '''Number of Bboxes.'''
        return self.bboxes.shape[0]

    def to_poly(self):
        '''Output the Bboxes polygons (list[list[np.ndarry]]).'''
        return [[p] for p in self.bboxes]

    @classmethod
    def from_poly(cls, polys):
        '''Create a Bbox instance from polygons (list[list[np.ndarray]]).'''
        if not polys:
            return cls.gen_empty()

        pts = []
        for poly in polys:
            _pts = np.concatenate(poly).reshape(-1, 2)
            pts.append(_pts.mean(axis=0))
        return PT(np.stack(pts, axis=0))

    def visualize(self, ax, texts, colors, thickness=1., font_size=10):
        '''see :func:`BaseBbox.visualize`'''
        num = len(self)
        assert len(colors) == len(texts) == num

        offset, pts = 3, self.bboxes
        for text, color, pt in zip(texts, colors, pts):
            x, y = pt
            if text:
                ax.text(x+25/2*thickness,
                        y,
                        text,
                        bbox={
                            'alpha': 0.5,
                            'pad': 0.7,
                            'facecolor': color,
                            'edgecolor': 'none'
                        },
                        color='white',
                        fontsize=font_size,
                        verticalalignment='center',
                        horizontalalignment='left')

        ax.scatter(pts[:, 0], pts[:, 1], s=25*thickness, c=colors, marker='o')

    @classmethod
    def concatenate(cls, bboxes):
        '''Concatenate list of bboxes.'''
        bboxes = []
        for b in bboxes:
            assert isinstance(b, PT)
            bboxes.append(b.bboxes)
        return PT(np.concatenate(bboxes, axis=0))

    def copy(self):
        '''Copy this instance.'''
        return PT(self.bboxes)

    @classmethod
    def gen_empty(cls):
        '''Create a Bbox instance with len == 0.'''
        bboxes = np.zeros((0, 2), dtype=np.float32)
        return PT(bboxes)

    def areas(self):
        '''ndarry: areas of each instance.'''
        return np.zeros((len(self), ), dtype=np.float32)

    def rotate(self, x, y, angle, keep_btype=True):
        '''see :func:`BaseBbox.rotate`'''
        M = cv2.getRotationMatrix2D((x, ), angle, 1)
        pts = self.bboxes[:, None, :]
        pts =cv2.transform(pts, M)

        output = PT(pts[:, 0, :])
        if not keep_btype:
            output = output.to_type(POLY)
        return output

    def warp(self, M, keep_btype=False):
        '''see :func:`BaseBbox.warp`'''
        pts = self.bboxes[:, None, :]
        if M.shape[0] == 2:
            pts = cv2.transform(pts, M)
        elif M.shape[0] == 3:
            pts = cv2.prospectiveTransform(pts, M)
        else:
            raise ValueError(f'Wrong M shape {M.shape}')

        output = PT(pts[:, 0, :])
        if not keep_btype:
            output = output.to_type(POLY)
        return output

    def flip(self, W, H, direction='horizontal'):
        '''see :func:`BaseBbox.flip`'''
        assert direction in ['horizontal', 'vertical', 'diagonal']

        output = self.copy()
        if direction == 'horizontal':
            output.bboxes[:, 0] = W - output.bboxes[:, 0]
        elif direction == 'vertical':
            output.bboxes[:, 1] = H - output.bboxes[:, 1]
        else:
            output.bboxes[:, 0] = W - output.bboxes[:, 0]
            output.bboxes[:, 1] = H - output.bboxes[:, 1]
        return output

    def translate(self, x, y):
        '''see :func:`BaseBbox.translate`'''
        output = self.copy()
        output.bboxes += np.asarray([x, y], dtype=np.float32)
        return output

    def resize(self, ratios):
        '''see :func:`BaseBbox.resize`'''
        output = self.copy()
        if isinstance(ratios, (tuple, list)):
            assert len(ratios) == 2
            output.bboxes *= np.asarray(ratios, dtype=np.float32)
        else:
            output.bboxes *= ratios
        return output
