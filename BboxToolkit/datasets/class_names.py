import os.path as osp


def voc_classes():
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]


def dota1_0_classes():
    return [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank',  'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]


def dota1_5_classes():
    return [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank',  'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter',
        'container-crane'
    ]


def dota2_0_classes():
    return [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank',  'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter',
        'container-crane', 'airport', 'helipad'
    ]


def dior_classes():
    return [
        'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'expressway-service-area', 'expressway-toll-station',
        'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
        'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
        'windmill'
    ]


def hrsc_classes():
    return ['ship']


def tinyperson_classes():
    return ['person']


dataset_aliases = {
    voc_classes: ['voc', 'pascal_voc', 'voc07', 'voc12'],
    coco_classes: ['coco', 'mscoco', 'ms_coco'],
    dota1_0_classes: ['DOTA1_0', 'DOTA1', 'dota1_0', 'dota1'],
    dota1_5_classes: ['DOTA1_5', 'DOTA1.5', 'dota1_5', 'dota1.5'],
    dota2_0_classes: ['DOTA2_0', 'DOTA2', 'dota2_0', 'dota2'],
    dior_classes: ['DIOR', 'dior'],
    hrsc_classes: ['HRSC', 'hrsc'],
    tinyperson_classes: ['TinyPerson', 'tinyperson']
}


def get_classes(alias_or_list):
    if isinstance(alias_or_list, (list, tuple)):
        for name in alias_or_list:
            assert isinstance(name, str)
        return alias_or_list

    elif isinstance(alias_or_list, str):
        if osp.isfile(alias_or_list):
            with open(alias_or_list, 'r') as f:
                classes = [line.strip() for line in f]
            return classes

        for func, v in dataset_aliases.items():
            if alias_or_list in v:
                return func()

        raise ValueError(f"Can't interpret dataset {alias_or_list}.")

    else:
        raise TypeError(f'dataset must a list, tuple or str, ',
                        f'but got {type(alias_or_list)}.')
