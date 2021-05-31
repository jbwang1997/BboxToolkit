import cv2
import inspect
import numpy as np


class BaseBbox:

    Bdim = None

    def __init__(self, bboxes, scores=None):
        assert self.Bdim is not None
        dim = bboxes.shape[-1]
        bboxes = bboxes.astype(np.float32)
        scores = scores if scores is None else \
                scores.astype(np.float32)

        if dim == self.Bdim:
            if scores is not None:
                assert bboxes.ndim == scores.ndim + 1
            self.bboxes = bboxes
            self.scores = scores
        elif dim == self.Bdim+1:
            assert scores is None
            self.bboxes = bboxes[..., :-1]
            self.scores = bboxes[..., -1]
        else:
            raise ValueError(f'The dim of {type(self).__name__} '
                             f'need to be {self.Bdim} or {self.Bdim+1}, '
                             f'but get {dim}')


    ## special functions for instance
    def copy(self):
        return type(self)(self.bboxes, self.scores)

    def flatten(self):
        dim = self.bboxes.shape[-1]
        return self.reshape((-1, dim))

    def reshape(self, new_shape):
        bboxes = self.bboxes
        new_bboxes = np.reshape(bboxes, new_shape)
        assert new_bboxes.shape[-1] == bboxes.shape[-1], \
                "reshape function can not change bboxes last dimension"
        new_scores = np.reshape(self.scores, new_shape[:-1]) \
                if self.scores is not None else None
        return type(self)(new_bboxes, new_scores)

    @property
    def shape(self):
        return self.bboxes.shape

    @property
    def is_empty(self):
        return self.bboxes.numel() == 0

    @property
    def with_scores(self):
        return self.scores is not None

    def __getitem__(self, index):
        bboxes = self.bboxes
        scores = self.scores
        if isinstance(index, tuple):
            assert len(index) < bboxes.ndim
        elif isinstance(index, np.ndarray):
            assert index.ndim < bboxes.ndim
        elif isinstance(index, int):
            assert bboxes.ndim > 2

        bboxes = bboxes[index]
        scores = None if scores is None else scores[index]
        return type(self)(bboxes, scores)

    def __repr__(self):
        repr_str = type(self).__name__
        repr_str += f'(bboxes={self.bboxes}, '
        repr_str += f'scores={self.scores})'
        return repr_str


    ## functions about transformation
    TRAN_SHORTCUTS = dict()

    @classmethod
    def register_tran_shortcuts(cls, start, end, force=False):
        assert isinstance(start, str) or inspect.isclass(start)
        if inspect.isclass(start):
            start = start.__name__
        start = start.lower()

        assert isinstance(end, str) or inspect.isclass(end)
        if inspect.isclass(end):
            end = end.__name__
        end = end.lower()

        key = start + '2' + end
        if (not force) and (key in cls.TRAN_SHORTCUTS):
            raise KeyError(f'The shortcut from {start} to {end} '
                           f'is already registered.')

        def _decorator(func):
            cls.TRAN_SHORTCUTS[key] = func
            return func

        return _decorator

    @classmethod
    def from_poly(cls, align_polys):
        bboxes = align_polys.bboxes
        return cls(cls.bbox_from_poly(bboxes), align_polys.scores)

    def to_poly(self):
        bboxes = self.bboxes
        return AlignPOLY(self.bbox_to_poly(bboxes), self.scores)

    def to_type(self, new_type):
        if not inspect.isclass(new_type):
            raise TypeError('new_type need to be a class')

        shortcut_name = type(self).__name__.lower() + '2' \
                + new_type.__name__.lower()
        if shortcut_name in self.TRAN_SHORTCUTS:
            return self.TRAN_SHORTCUTS[shortcut_name](self)

        align_polys = self.to_poly()
        return new_type.from_poly(align_polys)


    ## functions for distortion
    def roatate(self, center, angle):
        M = cv2.getRotationMatrix2D(center, angle, 1)
        return self.warp(M)

    def warp(self, M):
        polys = self.bbox_to_poly(self.bboxes)
        shape = polys.shape

        group_pts = polys.reshape(*shape[:-1], shape[-1]//2, 2)
        warped_pts = self.warp_pts(group_pts, M)
        warped_polys = warped_pts.reshape(*shape)

        new_bboxes = self.bbox_from_poly(warped_polys)
        return type(self)(new_bboxes, self.scores)

    @staticmethod
    def warp_pts(pts, M):
        pts = np.insert(pts, 2, 1, axis=-1)
        warped_pts = np.matmul(pts, M.T)
        if M.shape[0] == 3:
            warped_pts = (warped_pts/warped_pts[..., -1:])[..., :-1]
        return warped_pts


    ## functions need to be implemented by subclasses
    @classmethod
    def gen_empty(cls, with_scores=False):
        raise NotImplementedError

    @classmethod
    def gen_random(cls, shape, scale=1, with_scores=False):
        raise NotImplementedError

    @staticmethod
    def bbox_from_poly(polys):
        raise NotImplementedError

    @staticmethod
    def bbox_to_poly(bboxes):
        raise NotImplementedError

    def areas(self):
        raise NotImplementedError

    def flip(self, W, H, direction='horizontal'):
        raise NotImplementedError

    def translate(self, x, y):
        raise NotImplementedError

    def rescale(self, scales):
        raise NotImplementedError


class AlignPOLY(BaseBbox):

    def __init__(self, bboxes, scores=None):
        dim = bboxes.shape[-1]
        assert dim >= 2
        bboxes = bboxes.astype(np.float32)
        scores = scores if scores is None else \
                scores.astype(np.float32)

        if dim % 2 == 0:
            self.bboxes = bboxes
            self.scores = scores
        else:
            assert scores is None
            self.bboxes = bboxes[..., :-1]
            self.scores = bboxes[..., -1]

    @staticmethod
    def bbox_from_poly(polys):
        return polys

    @staticmethod
    def bbox_to_poly(bboxes):
        return bboxes

    def areas(self):
        bboxes = self.bboxes
        bboxes_ = np.concatenate(
            [bboxes[..., 2:], bboxes[..., :2]], dim=-1)

        areas = bboxes[..., 0::2] * bboxes_[..., 1::2] - \
                bboxes_[..., 0::2] * bboxes[..., 1::2]
        return np.abs(0.5 * areas.sum(dim=-1))

    def flip(self, W, H, direction='horizontal'):
        assert direction in ['horizontal', 'vertical', 'diagonal']
        bboxes = self.bboxes

        flipped = bboxes.copy()
        if direction == 'horizontal':
            flipped[..., 0::2] = W - bboxes[..., 0::2]
        elif direction == 'vertical':
            flipped[..., 1::2] = H - bboxes[..., 1::2]
        else:
            flipped[..., 0::2] = W - bboxes[..., 0::2]
            flipped[..., 1::2] = H - bboxes[..., 1::2]

        return type(self)(flipped, self.scores)

    def translate(self, x, y):
        dim = self.bboxes.shape[-1]
        new_bboxes = self.bboxes + np.array(
            (x, y) * (dim//2), dtype=np.float32)
        return type(self)(new_bboxes, self.scores)

    def rescale(self, scales):
        if isinstance(scales, (tuple, list)):
            assert len(scales) == 2
            dim = self.bboxes.shape[-1]
            new_bboxes = self.bboxes * np.array(
                scales * (dim//2), dtype=np.float32)
        else:
            new_bboxes = self.bboxes * scales
        return type(self)(new_bboxes, self.scores)
