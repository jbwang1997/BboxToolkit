_base_ = ['../_base_/dota_1s.py']

skip_empty = False
dataset_type = 'dota'
loading_args = dict(
    img_dir='path/to/dota/images',
    ann_dir='path/to/dota/annotations',
    nproc=1
)
dumping_dir = 'path/to/dump/splitted/annotations'
