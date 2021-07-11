import numpy as np
import shapely.geometry as shgeo

from ..structures import BaseBbox, HBB

EPS = 1e-4


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    assert mode in ['iou', 'iof']
    assert isinstance(bboxes1, BaseBbox)
    assert isinstance(bboxes2, BaseBbox)

    rows, cols = len(bboxes1), len(bboxes2)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        return np.zeros((rows, ), dtype=np.float32) if is_aligned \
                else np.zeros((rows, cols), dtype=np.float32)

    np_hbb1 = bboxes1.to_type('hbb').bboxes
    np_hbb2 = bboxes2.to_type('hbb').bboxes
    if not is_aligned:
        np_hbb1 = np_hbb1[:, None, :]

    lt = np.maximum(np_hbb1[..., :2], np_hbb2[..., :2])
    rb = np.minimum(np_hbb1[..., 2:], np_hbb2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    if isinstance(bboxes1, HBB) and isinstance(bboxes2, HBB):
        areas1 = (np_hbb1[..., 2] - np_hbb1[..., 0]) * (
            np_hbb1[..., 3] - np_hbb1[..., 1])
        if mode == 'iou':
            areas2 = (np_hbb2[..., 2] - np_hbb2[..., 0]) * (
                np_hbb2[..., 3] - np_hbb2[..., 1])
            unions = areas1 + areas2 - h_overlaps
            unions = np.maximum(unions, EPS)
        else:
            unions = np.maximum(areas1, EPS)
        return h_overlaps / unions
    else:
        areas1 = bboxes1.areas()
        if not is_aligned:
            areas1 = areas1[:, None]

        sg_polys1 = []
        for p in bboxes1:
            if len(p) > 1:
                p = [(p_.reshape(-1, 2), None) for p_ in p]
                sg_polys1.append(shgeo.MultiPolygon(p))
            else:
                sg_polys1.append(shgeo.Polygon(p[0].reshape(-1, 2)))

        sg_polys2 = []
        for p in bboxes2:
            if len(p) > 1:
                p = [(p_.reshape(-1, 2), None) for p_ in p]
                sg_polys2.append(shgeo.MultiPolygon(p))
            else:
                sg_polys2.append(shgeo.Polygon(p[0].reshape(-1, 2)))

        overlaps = np.zeros(h_overlaps.shape, dtype=np.float32)
        for p in zip(*np.nonzero(h_overlaps)):
            overlaps[p] = sg_polys1[p[0]].intersection(
                sg_polys2[p[-1]]).area

        if mode == 'iou':
            areas2 = bboxes2.areas()
            unions = np.maximum(areas1 + areas2 - overlaps, EPS)
        else:
            unions = np.maximum(areas1, EPS)
        return overlaps / unions
