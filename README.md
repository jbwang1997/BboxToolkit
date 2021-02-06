## Introduction
BboxToolkit is a tiny library including common operations for HBB(horizontal bboxes), OBB(Oriented bboxes) and POLY(poly bboxes).
The whole BboxToolkit is written by python and not requires compiling.
So it's easy to install BboxToolkit and use it in other projects. 
We have already implemented some tools using BboxToolkit for ease of use.

## License
This project is released under the [Apache 2.0 license](LICENSE)

## Dependencies
BboxToolkit will automatically install dependencies when you install, so this section is mostly for your reference.

BboxToolkit requires following dependencies:

+ Python > 3
+ Numpy
+ Opencv-python
+ Shapely
+ Terminaltables
+ Pillow

## Installation
BboxToolkit is written by python, no compilation is required.

```
git clone https://github.com/jbwang1997/BboxToolkit
cd BboxToolkit

python setup.py develop
```

## Usage
### Definition
There are three type of bboxes defining in BboxToolkit.

![bboxes define](definition.png)

HBB is denoted by the left-top point and right-bottom point like most detection dataset.

OBB is denoted by center point(x, y), width(w), height(h) and theta.
width is the longer side, height is the shorter side, theta is the angle between width and x-axis, we define clockwise as positive.
So, there must be w>h and theta∈[-pi/2, pi/2).

POLY is denoted by four points.
The order of these points doesn't matter, but the adjacent points should be a side of POLY.

### Function API
We will complish API documents in late update.

## Ackonwledgement
BboxToolkit refers to [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).

[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) is the toolbox for [DOTA](https://arxiv.org/abs/1711.10398) Dataset

[MMCV](https://github.com/open-mmlab/mmcv) is a foundational python library for computer vision.

[MMDetection](https://github.com/open-mmlab/mmdetection) is an open source object detection toolbox based on PyTorch.