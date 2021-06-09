import cv2
import numpy as np
import os.path as osp

from PIL import Image, ImageDraw, ImageFont
from ..misc import img_exts
from ..structures import BaseBbox


class DrawPipeline:

    def open_img(self, img):
        if isinstance(img, np.ndarray):
            img = img.ascontiguousarray(img)
            img = cv2.cvtColor(img, cv2.BGR2RGB)
            img = Image.fromarray(img)
        elif isinstance(img, str):
            assert osp.exists(img), f'{img} is not exists!'
            assert osp.splitext(img)[-1] in img_exts, \
                    f'{img} is not a image'
            img = Image.open(img)
        return img

    def pre_bboxes(self, bboxes, labels):
        if isinstance(bboxes, BaseBbox):
            if labels is None:
                new_bboxes = [bboxes.flatten()]
            else:
                new_bboxes = []
                for i in range(labels.max()):
                    new_bboxes.append(bboxes[labels==i].flatten())
        elif isinstance(bboxes, (list, tuple)):
            assert labels is None
            new_bboxes = [B.flatten() for B in bboxes]
        return new_bboxes

    def __call__(self, img, bboxes, labels=None):
        img = self.open_img(img)
        draw = ImageDraw.Draw(img)
        bboxes = self.pre_bboxes(bboxes, labels)

        for cls_id, B in enumerate(bboxes):
            for bbox_id in range(B.shape[0]):
                pass


def draw_bboxes(bboxes):
    pass


def draw_text(bboxes):
    pass
