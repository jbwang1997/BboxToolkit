import numpy as np
import shapely.geometry as shgeo

from ..structures import BaseBbox, HBB


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    assert mode in ['iou', 'iof']
    assert isinstance(bboxes1, BaseBbox)
    assert isinstance(bboxes2, BaseBbox)

    rows, cols = len(bboxes1), len(bboxes1)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        return np.zeros((rows, 1), dtype=np.float32) if is_aligned \
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
            return h_overlaps / (areas1 + areas2 - h_overlaps)
        else:
            return h_overlaps / areas1
    else:
        areas1 = bboxes1.areas()
        areas2 = bboxes2.areas()
        if not is_aligned:
            areas1 = areas1[None, :]

        sg_polys1 = []
        for p in bboxes1:
            if len(p) >= 1:
                p = [p_.reshape(-1, 2) for p_ in p]
                sg_polys1.append(shgeo.MultiPolygon(p))
            else:
                sg_polys1.append(shgeo.Polygon(p[0].reshape(-1, 2)))

        sg_polys2 = []
        for p in bboxes2:
            if len(p) >= 1:
                p = [p_.reshape(-1, 2) for p_ in p]
                sg_polys2.append(shgeo.MultiPolygon(p))
            else:
                sg_polys2.append(shgeo.Polygon(p[0].reshape(-1, 2)))

        overlaps = np.zeros(h_overlaps.shape, dtype=np.float32)
        for p in zip(*np.nonzero(h_overlaps)):
            overlaps[p] = sg_polys1[p[0]].intersection(
                sg_polys2[p[-1]]).area

        if mode == 'iou':
            return overlaps / (areas1 + areas2 - overlaps)
        else:
            return overlaps / areas1
