import os
import time
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np

from PIL import Image
from functools import partial
from multiprocessing import Pool
from .misc import img_exts


def load_hrsc(img_dir, ann_dir, classes=None, img_keys=dict(),
              obj_keys=dict(), nproc=10):
    if classes is not None:
        print('load_hrsc loads all objects as ship, arguments classes is no use')
    assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'
    assert ann_dir is None or osp.isdir(ann_dir), f'The {ann_dir} is not an existing dir!'

    contents = []
    print('Starting loading HRSC dataset information.')
    start_time = time.time()
    _load_func = partial(_load_hrsc_single,
                         img_dir=img_dir,
                         ann_dir=ann_dir,
                         img_keys=img_keys,
                         obj_keys=obj_keys)
    if nproc > 1:
        pool = Pool(nproc)
        contents = pool.map(_load_func, os.listdir(img_dir))
        pool.close()
    else:
        contents = list(map(_load_func, os.listdir(img_dir)))
    contents = [c for c in contents if c is not None]
    end_time = time.time()
    print(f'Finishing loading HRSC, get {len(contents)} images,',
          f'using {end_time-start_time:.3f}s.')
    return contents, ['ship']


def _load_hrsc_single(imgfile, img_dir, ann_dir, img_keys, obj_keys):
    img_id, ext = osp.splitext(imgfile)
    if ext not in img_exts:
        return None

    xmlfile = None if ann_dir is None else osp.join(ann_dir, img_id+'.xml')
    content = _load_hrsc_xml(xmlfile, img_keys, obj_keys)

    if not ('width' in content and 'height' in content):
        imgpath = osp.join(img_dir, imgfile)
        size = Image.open(imgpath).size
        content.update(dict(width=size[0], height=size[1]))
    content.update(dict(filename=imgfile, id=img_id))
    return content


def _load_hrsc_xml(xmlfile, img_keys=dict(), obj_keys=dict()):
    hbboxes, bboxes, diffs = list(), list(), list()
    content = {k: None for k in img_keys}
    ann = {k: [] for k in obj_keys}
    if xmlfile is None:
        pass
    elif not osp.isfile(xmlfile):
        print(f"Can't find {xmlfile}, treated as empty xmlfile")
    else:
        tree = ET.parse(xmlfile)
        root = tree.getroot()

        content['width'] = int(root.find('Img_SizeWidth').text)
        content['height'] = int(root.find('Img_SizeHeight').text)
        for k, xml_k in img_keys.items():
            node = root.find(xml_k)
            value = None if node is None else node.text
            content[k] = value

        objects = root.find('HRSC_Objects')
        for obj in objects.findall('HRSC_Object'):
            hbboxes.append([
                float(obj.find('box_xmin').text),
                float(obj.find('box_ymin').text),
                float(obj.find('box_xmax').text),
                float(obj.find('box_ymax').text)
            ])
            bboxes.append([
                float(obj.find('mbox_cx').text),
                float(obj.find('mbox_cy').text),
                float(obj.find('mbox_w').text),
                float(obj.find('mbox_h').text),
                -float(obj.find('mbox_ang').text)
            ])
            diffs.append(
                int(obj.find('difficult').text))

            for k, xml_k in obj_keys.items():
                node = obj.find(xml_k)
                value = None if node is None else node.text
                ann[k].append(value)

    hbboxes = np.array(hbboxes, dtype=np.float32) if hbboxes \
            else np.zeros((0, 4), dtype=np.float32)
    bboxes = np.array(bboxes, dtype=np.float32) if bboxes \
            else np.zeros((0, 5), dtype=np.float32)
    diffs = np.array(diffs, dtype=np.int64) if diffs \
            else np.zeros((0, ), dtype=np.int64)
    labels = np.zeros((bboxes.shape[0], ), dtype=np.int64)

    ann['hbboxes'] = hbboxes
    ann['bboxes'] = bboxes
    ann['diffs'] = diffs
    ann['labels'] = labels
    content['ann'] = ann
    return content
