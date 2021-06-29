import cv2
import numpy as np
from ..utils import pi

# Import Bbox type
from .base import BaseBbox
from .hbb import HBB
from .obb import OBB
from .p4poly import P4POLY
from .poly import POLY
from .pt import PT


#--------------------------HBB----------------------------
#-------------Start from HBB, End to others.--------------
@BaseBbox.register_shortcuts(HBB, OBB)
def HBB2OBB(hbbs):
    hbbs = hbbs.bboxes
    x = (hbbs[:, 0] + hbbs[:, 2]) * 0.5
    y = (hbbs[:, 1] + hbbs[:, 3]) * 0.5
    w = hbbs[..., 2] - hbbs[..., 0]
    h = hbbs[..., 3] - hbbs[..., 1]
    t = np.zeros((hbbs.shape[0], ),  dtype=np.float32)
    return OBB(np.stack([x, y, w, h, t], axis=1))


@BaseBbox.register_shortcuts(HBB, P4POLY)
def HBB2P4POLY(hbbs):
    l, t, r, b = np.split(hbbs.bboxes, 4, axis=1)
    return P4POLY(np.stack([l, t, r, t, r, b, l, b], axis=1))


@BaseBbox.register_shortcuts(HBB, POLY)
def HBB2POLY(hbbs):
    hbbs = hbbs.bboxes
    l, t, r, b = np.split(hbbs, 4, axis=1)
    p4polys = np.stack([l, t, r, t, r, b, l, b], axis=1)

    num = hbbs.shape[0]
    polys = POLY.gen_empty()
    polys.pts = p4polys.reshape(-1, 2)
    polys.regs = np.zeros((4*num, ), dtype=np.int64)
    polys.objs = np.arange(num, dtype=np.int64).repeat(4)
    return polys


@BaseBbox.register_shortcuts(HBB, PT)
def HBB2PT(hbbs):
    hbbs = hbbs.bboxes
    return PT(hbbs[:, 2:] - hbbs[:, :2])


#--------------------------OBB----------------------------
#-------------Start from OBB, End to others.--------------
@BaseBbox.register_shortcuts(OBB, HBB)
def OBB2HBB(obbs):
    ctr, w, h, theta = np.split(obbs.bboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias = np.abs(w/2 * Cos) + np.abs(h/2 * Sin)
    y_bias = np.abs(w/2 * Sin) + np.abs(h/2 * Cos)
    bias = np.concatenate([x_bias, y_bias], axis=-1)
    return HBB(np.concatenate([ctr-bias, ctr+bias], axis=1))


@BaseBbox.register_shortcuts(OBB, P4POLY)
def OBB2P4POLY(obbs):
    ctr, w, h, theta = np.split(obbs.bboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)

    vec1 = np.concatenate(
        [w/2 * Cos, w/2 * Sin], axis=1)
    vec2 = np.concatenate(
        [-h/2 * Sin, h/2 * Cos], axis=1)

    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return P4POLY(np.concatenate([pt1, pt2, pt3, pt4], axis=-1))


@BaseBbox.register_shortcuts(OBB, POLY)
def OBB2POLY(obbs):
    ctr, w, h, theta = np.split(obbs.bboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)

    vec1 = np.concatenate(
        [w/2 * Cos, w/2 * Sin], axis=1)
    vec2 = np.concatenate(
        [-h/2 * Sin, h/2 * Cos], axis=1)

    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2

    polys = POLY.gen_empty()
    polys.pts = np.stack([pt1, pt2, pt3, pt4], axis=1).reshape(-1, 2)
    polys.regs = np.zeros((polys.pts.shape[0], ), dtype=np.int64)
    polys.objs = np.arange(len(obbs), dtype=np.int64).repeat(4)
    return polys


@BaseBbox.register_shortcuts(OBB, PT)
def OBB2PT(obbs):
    return PT(obbs.bboxes[:, :2])


#------------------------P4POLY---------------------------
#------------Start from P4POLY, End to others.------------
@BaseBbox.register_shortcuts(P4POLY, HBB)
def P4POLY2HBB(p4polys):
    p4polys = p4polys.bboxes
    p4polys = p4polys.reshape(-1, 4, 2)
    lt_points = np.min(p4polys, axis=1)
    rb_points = np.max(p4polys, axis=1)
    return HBB(np.concatenate([lt_points, rb_points], axis=1))


@BaseBbox.register_shortcuts(P4POLY, OBB)
def P4POLY2OBB(p4polys):
    p4polys = p4polys.bboxes.reshape(-1, 4, 2)
    obboxes = []
    for poly in p4polys:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        obboxes.append([x, y, w, h, angle/180*pi])
    return OBB(np.asarray(obboxes, dtype=np.float32))


@BaseBbox.register_shortcuts(P4POLY, POLY)
def P4POLY2POLY(p4polys):
    polys = POLY.gen_empty()
    polys.pts = p4polys.bboxes.reshape(-1, 2)
    polys.regs = np.zeros((polys.pts.shape[0], ), dtype=np.int64)
    polys.objs = np.arange(len(p4polys), dtype=np.int64).repeat(4)
    return polys


@BaseBbox.register_shortcuts(P4POLY, PT)
def P4POLY2PT(p4polys):
    p4polys.p4polys.bboxes.reshape(-1, 4, 2)
    return PT(p4polys.mean(axis=1))


#-------------------------POLY----------------------------
#-------------Start from POLY, End to others.-------------
@BaseBbox.register_shortcuts(POLY, HBB)
def POLY2HBB(polys):
    if len(polys) == 0:
        return HBB.gen_empty()
    
    pts = polys.pts
    objs = polys.objs
    hbbs = []
    for i in range(len(polys)):
        _pts = pts[objs == i]
        lt_points = _pts.min(axis=0)
        rb_points = _pts.max(axis=0)
        hbbs.append(np.concatenate(
            [lt_points, rb_points], axis=0))
    return HBB(np.stack(hbbs, axis=0))


@BaseBbox.register_shortcuts(POLY, OBB)
def POLY2OBB(polys):
    if len(polys) == 0:
        return OBB.gen_empty()

    pts = polys.pts
    objs = polys.objs
    obbs = []
    for i in range(len(polys)):
        _pts = pts[objs == i]
        (x, y), (w, h), angle = cv2.minAreaRect(_pts)
        obbs.append([x, y, w, h, angle/180*pi])
    return OBB(np.asarray(obbs))


@BaseBbox.register_shortcuts(POLY, P4POLY)
def POLY2P4POLY(polys):
    if len(polys) == 0:
        return OBB.gen_empty()

    pts = polys.pts
    regs = polys.regs
    objs = polys.objs
    p4poly = []
    for i in range(len(polys)):
        _pts = pts[objs == i]
        _regs = regs[objs == i]
        if _pts.shape[0] == 4 and np.all(_regs == 0):
            p4poly.append(_pts)
        else:
            p4poly.append(cv2.boxPoints(cv2.minAreaRect(_pts)))
    p4poly = np.stack(p4poly, axis=0)
    return P4POLY(p4poly.reshape(-1, 8))


@BaseBbox.register_shortcuts(POLY, PT)
def POLY2PT(polys):
    if len(polys) == 0:
        return PT.gen_empty()

    pts = polys.pts
    objs = polys.objs
    ctr_pts = []
    for i in range(len(polys)):
        _pts = pts[objs == i]
        ctr_pts.append(_pts.mean(axis=0))
    return PT(np.stack(ctr_pts, axis=0))


#--------------------------PT-----------------------------
#--------------Start from PT, End to others.--------------
@BaseBbox.register_shortcuts(PT, HBB)
def PT2HBB(pts):
    pts = pts.bboxes
    return HBB(np.concatenate([pts, pts], axis=1))


@BaseBbox.register_shortcuts(PT, OBB)
def PT2OBB(pts):
    pts = pts.bboxes
    wht = np.zeros((pts.shape[0], 3), dtype=np.float32)
    obbs = np.concatenate([pts, wht], axis=1)
    return OBB(obbs)


@BaseBbox.register_shortcuts(PT, P4POLY)
def PT2P4POLY(pts):
    pts = pts.bboxes
    return P4POLY(pts.tile(pts, 4, axis=1))


@BaseBbox.register_shortcuts(PT, POLY)
def PT2POLY(pts):
    polys = POLY.gen_empty()
    polys.pts = pts.bboxes
    polys.regs = np.zeros((len(pts), ), dtype=np.int64)
    polys.objs = np.arange(len(pts), dtype=np.int64)
    return polys
