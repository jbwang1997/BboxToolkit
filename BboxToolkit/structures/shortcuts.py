'''Direct transformation functions. All functions are registered into
BaseBbox, so they can be run by function BaseBbox.to_type.

Please refer to BaseBbox.to_type to get details.
'''
import cv2
import numpy as np
from ..utils.defaults import pi

# Import Bbox type
from .base import BaseBbox
from .hbb import HBB
from .obb import OBB
from .p4poly import P4POLY
from .poly import POLY
from .pt import PT


#--------------------------HBB----------------------------
#-------------Start from HBB, End to others.--------------
@BaseBbox.register_shortcuts('hbb', 'obb')
def hbb_2_obb(hbbs):
    hbbs = hbbs.bboxes
    x = (hbbs[:, 0] + hbbs[:, 2]) * 0.5
    y = (hbbs[:, 1] + hbbs[:, 3]) * 0.5
    w = hbbs[..., 2] - hbbs[..., 0]
    h = hbbs[..., 3] - hbbs[..., 1]
    t = np.zeros((hbbs.shape[0], ),  dtype=np.float32)
    return OBB(np.stack([x, y, w, h, t], axis=1))


@BaseBbox.register_shortcuts('hbb', 'p4poly')
def hbb_2_p4poly(hbbs):
    l, t, r, b = np.split(hbbs.bboxes, 4, axis=1)
    return P4POLY(np.stack([l, t, r, t, r, b, l, b], axis=1))


@BaseBbox.register_shortcuts('hbb', 'poly')
def hbb_2_poly(hbbs):
    hbbs = hbbs.bboxes
    l, t, r, b = np.split(hbbs, 4, axis=1)
    p4polys = np.stack([l, t, r, t, r, b, l, b], axis=1)

    num = hbbs.shape[0]
    polys = POLY.gen_empty()
    polys.pts = p4polys.reshape(-1, 2)
    polys.regs = np.zeros((4*num, ), dtype=np.int64)
    polys.objs = np.arange(num, dtype=np.int64).repeat(4)
    return polys


@BaseBbox.register_shortcuts('hbb', 'pt')
def hbb_2_pt(hbbs):
    hbbs = hbbs.bboxes
    return PT(hbbs[:, 2:] - hbbs[:, :2])


#--------------------------OBB----------------------------
#-------------Start from OBB, End to others.--------------
@BaseBbox.register_shortcuts('obb', 'hbb')
def obb_2_hbb(obbs):
    ctr, w, h, theta = np.split(obbs.bboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias = np.abs(w/2 * Cos) + np.abs(h/2 * Sin)
    y_bias = np.abs(w/2 * Sin) + np.abs(h/2 * Cos)
    bias = np.concatenate([x_bias, y_bias], axis=-1)
    return HBB(np.concatenate([ctr-bias, ctr+bias], axis=1))


@BaseBbox.register_shortcuts('obb', 'p4poly')
def obb_2_p4poly(obbs):
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


@BaseBbox.register_shortcuts('obb', 'poly')
def obb_2_poly(obbs):
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


@BaseBbox.register_shortcuts('obb', 'pt')
def obb_2_pt(obbs):
    return PT(obbs.bboxes[:, :2])


#------------------------P4POLY---------------------------
#------------Start from P4POLY, End to others.------------
@BaseBbox.register_shortcuts('p4poly', 'hbb')
def p4poly_2_hbb(p4polys):
    p4polys = p4polys.bboxes
    p4polys = p4polys.reshape(-1, 4, 2)
    lt_points = np.min(p4polys, axis=1)
    rb_points = np.max(p4polys, axis=1)
    return HBB(np.concatenate([lt_points, rb_points], axis=1))


@BaseBbox.register_shortcuts('p4poly', 'obb')
def p4poly_2_obb(p4polys):
    p4polys = p4polys.bboxes.reshape(-1, 4, 2)
    obboxes = []
    for poly in p4polys:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        obboxes.append([x, y, w, h, angle/180*pi])
    return OBB(np.asarray(obboxes, dtype=np.float32))


@BaseBbox.register_shortcuts('p4poly', 'poly')
def p4poly_2_poly(p4polys):
    polys = POLY.gen_empty()
    polys.pts = p4polys.bboxes.reshape(-1, 2)
    polys.regs = np.zeros((polys.pts.shape[0], ), dtype=np.int64)
    polys.objs = np.arange(len(p4polys), dtype=np.int64).repeat(4)
    return polys


@BaseBbox.register_shortcuts('p4poly', 'pt')
def p4poly_2_pt(p4polys):
    p4polys.p4polys.bboxes.reshape(-1, 4, 2)
    return PT(p4polys.mean(axis=1))


#-------------------------POLY----------------------------
#-------------Start from POLY, End to others.-------------
@BaseBbox.register_shortcuts('poly', 'hbb')
def poly_2_hbb(polys):
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


@BaseBbox.register_shortcuts('poly', 'obb')
def poly_2_obb(polys):
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


@BaseBbox.register_shortcuts('poly', 'p4poly')
def poly2p4poly(polys):
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


@BaseBbox.register_shortcuts('poly', 'pt')
def poly_2_pt(polys):
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
@BaseBbox.register_shortcuts('pt', 'hbb')
def pt_2_hbb(pts):
    pts = pts.bboxes
    return HBB(np.concatenate([pts, pts], axis=1))


@BaseBbox.register_shortcuts('pt', 'obb')
def pt_2_obb(pts):
    pts = pts.bboxes
    wht = np.zeros((pts.shape[0], 3), dtype=np.float32)
    obbs = np.concatenate([pts, wht], axis=1)
    return OBB(obbs)


@BaseBbox.register_shortcuts('pt', 'p4poly')
def pt_2_p4poly(pts):
    pts = pts.bboxes
    return P4POLY(pts.tile(pts, 4, axis=1))


@BaseBbox.register_shortcuts('pt', 'poly')
def pt_2_poly(pts):
    polys = POLY.gen_empty()
    polys.pts = pts.bboxes
    polys.regs = np.zeros((len(pts), ), dtype=np.int64)
    polys.objs = np.arange(len(pts), dtype=np.int64)
    return polys
