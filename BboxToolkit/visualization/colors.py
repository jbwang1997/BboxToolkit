# Refer to mmcv: https://github.com/open-mmlab/mmcv

import cv2
import numpy as np
from enum import Enum


class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    cyan = (0, 255, 255)
    yellow = (255, 255, 0)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating RGB channels.
    """
    if isinstance(color, str):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert channel >= 0 and channel <= 255
        return color
    elif isinstance(color, int):
        assert color >= 0 and color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError('Invalid type for color: {}'.format(type(color)))


def color_rand(num):
    pass

