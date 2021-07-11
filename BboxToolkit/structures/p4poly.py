import cv2
import numpy as np

from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection

from .base import BaseBbox
from .poly import POLY

import pdb


@BaseBbox.register_bbox_cls()
class P4POLY(BaseBbox):
    '''
    4 Point Polygon (P4POLY): present a bbox with coordinates of 4 points.

    Args:
        bboxes (ndarray (n, 8)): contain the coordinates of 4 points.
    '''

    def __init__(self, bboxes):
        assert isinstance(bboxes, np.ndarray)
        assert bboxes.ndim == 2 and bboxes.shape[1] == 8

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
        return P4POLY(self.bboxes[index])

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

        p4polys = []
        for poly in polys:
            if len(poly) == 1 and poly[0].shape[0] == 8:
                p4polys.append(poly[0])
            else:
                pts = np.concatenate(poly).reshape(-1, 2)
                pts = cv2.boxPoints(cv2.minAreaRect(pts))
                p4polys.append(pts.reshape(-1))
        return P4POLY(np.stack(p4polys, axis=0))

    def visualize(self, ax, texts, colors, thickness=1., font_size=10):
        '''see :func:`BaseBbox.visualize`'''
        num = len(self)
        assert len(colors) == len(texts) == num

        p4polys = self.bboxes
        pts = p4polys.reshape(-1, 4, 2)
        top_pts_idx = np.argsort(pts[..., 1], axis=1)[:, :2]
        top_pts_idx = top_pts_idx[..., None].repeat(2, axis=2)
        top_pts = np.take_along_axis(pts, top_pts_idx, axis=1)

        x_sort_idx = np.argsort(top_pts[..., 0], axis=1)
        left_idx, right_idx = x_sort_idx[:, :1], x_sort_idx[:, 1:]
        left_idx = left_idx[..., None].repeat(2, axis=2)
        left_pts = np.take_along_axis(top_pts, left_idx, axis=1).squeeze(1)
        right_idx = right_idx[..., None].repeat(2, axis=2)
        right_pts = np.take_along_axis(top_pts, right_idx, axis=1).squeeze(1)

        x2 = right_pts[:, 1] - left_pts[:, 1]
        x1 = right_pts[:, 0] - left_pts[:, 0]
        angles = np.arctan2(x2, x1) / np.pi * 180

        patches, edge_colors = [], []
        for text, color, pt, anchor, angle in zip(
                texts, colors, pts, left_pts, angles):
            x, y = anchor
            if text:
                ax.text(x,
                        y,
                        text,
                        bbox={
                            'alpha': 0.5,
                            'pad': 0.7,
                            'facecolor': color,
                            'edgecolor': 'none'
                        },
                        color='white',
                        rotation=angle,
                        rotation_mode='anchor',
                        fontsize=font_size,
                        transform_rotates_text=True,
                        verticalalignment='bottom',
                        horizontalalignment='left')

            patches.append(Polygon(pt))
            edge_colors.append(color)

        p = PatchCollection(
            patches,
            facecolors='none',
            edgecolors=edge_colors,
            linewidths=thickness)
        ax.add_collection(p)

    @classmethod
    def concatenate(cls, bboxes):
        '''Concatenate list of bboxes.'''
        bboxes = []
        for b in bboxes:
            assert isinstance(b, P4POLY)
            bboxes.append(b.bboxes)
        return P4POLY(np.concatenate(bboxes, axis=0))

    def copy(self):
        '''Copy this instance.'''
        return P4POLY(self.bboxes)

    @classmethod
    def gen_empty(cls):
        '''Create a Bbox instance with len == 0.'''
        bboxes = np.zeros((0, 8), dtype=np.float32)
        return P4POLY(bboxes)

    def areas(self):
        '''ndarry: areas of each instance.'''
        pts = self.bboxes.reshape(-1, 4, 2)
        part1 = pts[:, :, 0] * np.roll(pts[:, :, 1], 1, axis=1)
        part2 = pts[:, :, 1] * np.roll(pts[:, :, 0], 1, axis=1)
        areas = 0.5 * np.abs(part1.sum(axis=1) - part2.sum(axis=1))
        return areas

    def rotate(self, x, y, angle, keep_btype=True):
        '''see :func:`BaseBbox.rotate`'''
        # List all points
        pts = self.bboxes.reshape(-1, 4, 2)

        # Get the roatation matrix and rotate all points
        M = cv2.getRotationMatrix2D((x, y), angle, 1)
        rotated_pts = cv2.transform(pts, M)

        # Convert points to Output.
        rotated_pts = rotated_pts.reshape(-1, 8)
        if keep_btype:
            return P4POLY(rotated_pts)
        else:
            return POLY([[p] for p in rotated_pts])

    def warp(self, M, keep_btype=False):
        '''see :func:`BaseBbox.warp`'''
        # List the points of HBBs.
        pts = self.bboxes.reshape(-1, 4, 2)

        # Warp points.
        if M.shape[0] == 2:
            warped_pts = cv2.transform(pts, M)
        elif M.shape[0] == 3:
            warped_pts = cv2.prospectiveTransform(pts, M)
        else:
            raise ValueError(f'Wrong M shape {M.shape}')

        # Transform points to Output.
        warped_pts = warped_pts.reshape(-1, 8)
        if keep_btype:
            return P4POLY(warped_pts)
        else:
            return POLY([[p] for p in warped_pts])

    def flip(self, W, H, direction='horizontal'):
        '''see :func:`BaseBbox.flip`'''
        assert direction in ['horizontal', 'vertical', 'diagonal']

        bboxes = self.bboxes.copy()
        if direction == 'horizontal':
            bboxes[:, 0::2] = W - bboxes[:, 0::2]
        elif direction == 'vertical':
            bboxes[:, 1::2] = H - bboxes[:, 1::2]
        else:
            bboxes[:, 0::2] = W - bboxes[:, 0::2]
            bboxes[:, 1::2] = H - bboxes[:, 1::2]

        return P4POLY(bboxes)

    def translate(self, x, y):
        '''see :func:`BaseBbox.translate`'''
        bboxes = self.bboxes + np.asarray(
            [x, y, x, y, x, y, x, y], dtype=np.float32)
        return P4POLY(bboxes)

    def resize(self, ratios):
        '''see :func:`BaseBbox.resize`'''
        if isinstance(ratios, (tuple, list)):
            assert len(ratios) == 2
            bboxes = self.bboxes * np.asarray(ratios * 4, dtype=np.float32)
        else:
            bboxes = self.bboxes * ratios
        return P4POLY(bboxes)
