import cv2
import numpy as np
from .base import BaseBbox


class POLY(BaseBbox):
    '''
    Polygons (POLY): Interpret and store the polygon of bboxes in points,
    region ids, and object ids.

    Args:
        bboxes (list[list[ndarray]]): The first level of the list corresponds
            to objects, the second level to the polys that compose the object,
            the third level to the poly coordinates.
    '''

    def __init__(self, bboxes):
        if not bboxes:
            # Empty case
            self.pts = np.zeros((0, 2), dtype=np.float32)
            self.regs = np.zeros((0, ), dtype=np.int64)
            self.objs = np.zeros((0, ), dtype=np.int64)
        else:
            # Interpret the polygons.
            pts, regs, objs = [], [], []
            for i_obj, regions in enumerate(bboxes):
                for i_reg, region in enumerate(regions):
                    region = region.reshape(-1, 2).astype(np.float32)
                    regs.append(np.full(
                        (region.shape[0], ), i_reg, dtype=np.int64))
                    objs.append(np.full(
                        (region.shape[0], ), i_obj, dtype=np.int64))
                    pts.append(region)

            self.pts = np.concatenate(pts, axis=0)
            self.regs = np.concatenate(regs, axis=0)
            self.objs = np.concatenate(objs, axis=0)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'len={len(self)}, '
        s += f'pts={self.pts}, '
        s += f'regs={self.regs}'
        s += f'objs={self.objs})'
        return s

    def __getitem__(self, index):
        '''see :func:`BaseBbox.__getitem__`'''
        # Convert bool ndarray to index.
        if isinstance(index, np.ndarray):
            assert index.ndim == 1
            if index.dtype.type is np.bool_:
                index = np.nonzero(index)[0]

        output = POLY([])
        if len(index) == 0:
            # Zero index.
            return output

        _pts = self.pts
        _regs = self.regs
        _objs = self.objs
        pts, regs, objs = [], [], []
        for i, idx in enumerate(index):
            mask = _objs == idx
            num_pts = mask.sum()
            pts.append(_pts[mask])
            regs.append(_regs[mask])
            objs.append(np.full((num_pts, ), i, dtype=np.int64))
        output.pts = np.concatenate(pts, axis=0)
        output.regs = np.concatenate(regs, axis=0)
        output.objs = np.concatenate(objs, axis=0)
        return output

    def __len__(self):
        '''Number of Bboxes.'''
        return 0 if self.objs.size == 0 else self.objs.max()

    def to_poly(self):
        '''Output the Bboxes polygons (list[list[np.ndarry]]).'''
        # Empty case
        if len(self) == 0:
            return []

        pts = self.pts
        regs = self.regs
        objs = self.objs
        polys = []
        for i_obj in range(objs.max()):
            _pts = pts[objs == i_obj]
            _regs = regs[objs == i_obj]
            polys.append([_pts[_regs == i].reshape(-1)
                          for i in range(_regs.max())])
        return polys

    @classmethod
    def from_poly(cls, polys):
        '''Create a Bbox instance from polygons (list[list[np.ndarray]]).'''
        return POLY(polys)

    @classmethod
    def concatenate(cls, bboxes):
        '''Concatenate list of bboxes.'''
        pts, regs, objs = [], [], []
        start_len = 0
        for b in bboxes:
            pts.append(b.pts)
            regs.append(b.regs)
            objs.append(b.objs+start_len)
            start_len += len(b)

        polys = POLY.gen_empty()
        polys.pts = np.concatenate(pts)
        polys.regs = np.concatenate(regs)
        polys.objs = np.concatenate(objs)
        return polys

    def copy(self):
        '''Copy this instance.'''
        polys = POLY([])
        polys.pts = self.pts.copy()
        polys.regs = self.regs.copy()
        polys.objs = self.objs.copy()
        return polys

    @classmethod
    def gen_empty(cls):
        '''Create a Bbox instance with len == 0.'''
        return POLY([])

    def areas(self):
        '''ndarry: areas of each instance.'''
        if len(self) == 0:
            return np.zeros((0, ), dtype=np.float32)

        pts = self.pts
        regs = self.regs
        objs = self.objs
        areas = []
        for i_obj in range(objs.max()):
            area = 0
            _pts = pts[objs == i_obj]
            _regs = regs[objs == i_obj]
            for i_reg in range(_regs.max()):
                reg_pts = _pts[regs == i_reg]
                area += 0.5 * np.abs(
                    np.dot(reg_pts[:, 0], np.roll(reg_pts[:, 1], 1)) -\
                    np.dot(reg_pts[:, 1], np.roll(reg_pts[:, 0], 1)))
            areas.append(area)
        return np.asarray(areas, dtype=np.float32)

    def rotate(self, x, y, angle, keep_btype=True):
        '''see :func:`BaseBbox.rotate`'''
        M = cv2.getRotationMatrix2D((x, y), angle, 1)
        pts = self.pts[:, None, :]
        pts = cv2.transform(pts, M)

        output = self.copy()
        output.pts = pts[:, 0, :]
        return output

    def warp(self, M, keep_btype=False):
        '''see :func:`BaseBbox.warp`'''
        pts = self.pts[:, None, :]
        if M.shape[0] == 2:
            pts = cv2.transform(pts, M)
        elif M.shape[0] == 3:
            pts = cv2.prospectiveTransform(pts, M)
        else:
            raise ValueError(f'Wrong M shape {M.shape}')

        output = self.copy()
        output.pts = pts[:, 0, :]
        return output

    def flip(self, W, H, direction='horizontal'):
        '''see :func:`BaseBbox.flip`'''
        assert direction in ['horizontal', 'vertical', 'diagonal']

        output = self.copy()
        if direction == 'horizontal':
            output.pts[:, 0] = W - output.pts[:, 0]
        elif direction == 'vertical':
            output.pts[:, 1] = H - output.pts[:, 1]
        else:
            output.pts[:, 0] = W - output.pts[:, 0]
            output.pts[:, 1] = H - output.pts[:, 1]
        return output

    def translate(self, x, y):
        '''see :func:`BaseBbox.translate`'''
        output = self.copy()
        output.pts += np.asarray([x, y], dtype=np.float32)
        return output

    def resize(self, ratios):
        '''see :func:`BaseBbox.resize`'''
        output = self.copy()
        if isinstance(ratios, (tuple, list)):
            assert len(ratios) == 2
            output.pts *= np.asarray(ratios, dtype=np.float32)
        else:
            output.pts *= ratios
        return output
