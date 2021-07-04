import numpy as np

from .overlaps import bbox_overlaps
from ..structures import BaseBbox


def bbox_nms(bboxes, scores, iou_thr=0.5):
    assert isinstance(bboxes, BaseBbox)
    order = scores.argsort()[::-1]
    bboxes = bboxes[order]
    ious = bbox_overlaps(bboxes, bboxes)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        ious_with_other = ious[0]
        idx = np.where(ious_with_other <= iou_thr)[0]
        order = order[idx]
        ious = ious[idx, idx]

    return np.array(keep, dtype=np.int64)
