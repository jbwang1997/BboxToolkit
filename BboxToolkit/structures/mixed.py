import numpy as np

from .base import BaseBbox
from .poly import POLY


@BaseBbox.register_bbox_cls()
class MixedBbox(BaseBbox):
    '''
    Mixed Bboxes (MixedBbox): Mix different types of bboxes into one type.

    Args :
        bboxes (list[obj:subclass of BaseBboxes]): A list of bboxes.
    '''

    def __init__(self, bboxes):
        bbox_list = []
        for _bboxes in bboxes:
            if isinstance(_bboxes, MixedBbox):
                bbox_list.extend(
                    [b.copy() for b in _bboxes.bboxes])
            else:
                assert isinstance(_bboxes, BaseBbox)
                bbox_list.append(_bboxes.copy())
        self.bboxes = bbox_list

    def to_type(self, new_type):
        '''see :func:`BaseBbox.to_type`'''
        bboxes = [b.to_type(new_type) for b in self.bboxes]
        return new_type.concatenate(bboxes)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'len={len(self)})'
        return s

    def __getitem__(self, index):
        '''see :func:`BaseBbox.__getitem__`'''
        raise NotImplementedError(
            'Cannot get items from MixedBbox by index.')

    def __len__(self):
        '''Number of Bboxes.'''
        return sum([len(b) for b in self.bboxes])

    def to_poly(self):
        '''Output the Bboxes polygons (list[list[np.ndarry]]).'''
        polys = []
        for _bboxes in self.bboxes:
            polys.extend(_bboxes.to_poly())
        return polys

    @classmethod
    def from_poly(cls, polys):
        '''Create a Bbox instance from polygons (list[list[np.ndarray]]).'''
        raise NotImplementedError(
            'MixedBbox cannot be constructed from polygons.')

    def visualize(self, ax, texts, colors, thickness=1., font_size=10):
        '''see :func:`BaseBbox.visualize`'''
        for _bboxes in self.bboxes:
            _bboxes.visualize(ax, texts, colors, thickness, font_size)

    @classmethod
    def concatenate(cls, bboxes):
        '''Concatenate list of bboxes.'''
        return MixedBbox(bboxes)

    def copy(self):
        '''Copy this instance.'''
        return MixedBbox(self.bboxes)

    @classmethod
    def gen_empty(cls):
        '''Create a Bbox instance with len == 0.'''
        return MixedBbox([])

    def areas(self):
        '''ndarry: areas of each instance.'''
        areas = [b.areas() for b in self.bboxes]
        return np.concatenate(areas, axis=0)

    def rotate(self, x, y, angle, keep_btype=True):
        '''see :func:`BaseBbox.rotate`'''
        bboxes = []
        for _bboxes in self.bboxes:
            bboxes.append(_bboxes.rotate(x, y, angle, keep_btype))

        if keep_btype:
            return MixedBbox(bboxes)
        else:
            return POLY.concatenate(bboxes)

    def warp(self, M, keep_btype=False):
        '''see :func:`BaseBbox.warp`'''
        bboxes = []
        for _bboxes in self.bboxes:
            bboxes.append(_bboxes.warp(M, keep_btype))

        if keep_btype:
            return MixedBbox(bboxes)
        else:
            return POLY.concatenate(bboxes)

    def flip(self, W, H, direction='horizontal'):
        '''see :func:`BaseBbox.flip`'''
        bboxes = []
        for _bboxes in self.bboxes:
            bboxes.append(_bboxes.flip(W, H, direction))
        return MixedBbox(bboxes)

    def translate(self, x, y):
        '''see :func:`BaseBbox.translate`'''
        bboxes = []
        for _bboxes in self.bboxes:
            bboxes.append(_bboxes.translate(x, y))
        return MixedBbox(bboxes)

    def resize(self, ratios):
        '''see :func:`BaseBbox.resize`'''
        bboxes = []
        for _bboxes in self.bboxes:
            bboxes.append(_bboxes.resize(ratios))
        return MixedBbox(bboxes)
