'''Collect parameters using in BboxToolkit.'''
import numpy as np


# The files with following extensions will be regarded as images
img_exts = ['.bmp', '.tif', '.jpg', '.png']


# The approximation of pi
pi = round(np.pi, 4)

def approx_pi(num):
    assert isinstance(num, int)

    global pi
    pi = round(np.pi, num)
