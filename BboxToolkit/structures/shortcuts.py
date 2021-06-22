import cv2
import numpy as np
from .base import BaseBbox
from .hbb import HBB
from .obb import OBB
from .poly import P4POLY

pi = np.pi


@BaseBbox.register_shortcuts(P4POLY, OBB)
def poly2obb(instance):
    polys = instance.bboxes
    order = polys.shape[:-1]
    polys = polys.reshape(-1, 4, 2)

    obboxes = []
    for poly in polys:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        obboxes.append([x, y, w, h, angle/180*pi])
    obboxes = np.array(obboxes, dtype=np.float32).reshape(*order, 5)
    return OBB(obboxes, instance.scores)


@BaseBbox.register_shortcuts(P4POLY, HBB)
def poly2hbb(instance):
    polys = instance.bboxes
    shape = polys.shape
    polys = polys.reshape(*shape[:-1], shape[-1]//2, 2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return HBB(np.concatenate([lt_point, rb_point], axis=-1),
               instance.scores)


@BaseBbox.register_shortcuts(OBB, P4POLY)
def obb2poly(instance):
    ctr, w, h, theta = np.split(instance.bboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)

    vec1 = np.concatenate(
        [w/2 * Cos, w/2 * Sin], axis=-1)
    vec2 = np.concatenate(
        [-h/2 * Sin, h/2 * Cos], axis=-1)

    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return P4POLY(np.concatenate([pt1, pt2, pt3, pt4], axis=-1),
                  instance.scores)


@BaseBbox.register_shortcuts(OBB, HBB)
def obb2hbb(instance):
    obboxes = instance.bboxes
    ctr, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias = np.abs(w/2 * Cos) + np.abs(h/2 * Sin)
    y_bias = np.abs(w/2 * Sin) + np.abs(h/2 * Cos)
    bias = np.concatenate([x_bias, y_bias], axis=-1)
    return HBB(np.concatenate([ctr-bias, ctr+bias], axis=-1),
               instance.scores)


@BaseBbox.register_shortcuts(HBB, P4POLY)
def hbb2poly(instance):
    hbboxes = instance.bboxes
    l, t, r, b = np.split(hbboxes, 4, axis=-1)
    return P4POLY(np.stack([l, t, r, t, r, b, l, b], axis=-1),
                  instance.scores)


@BaseBbox.register_shortcuts(HBB, OBB)
def hbb2obb(instance):
    hbboxes = instance.bboxes
    order = hbboxes.shape[:-1]
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    t = np.zeros(order, dtype=np.float32)
    return OBB(np.stack([x, y, w, h, t], axis=-1),
               instance.scores)
