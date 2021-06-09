import os
import numpy as np
import matplotlib.colors as mpt_colors


def list_named_colors(out_file=None):
    color_dict = mpt_colors.get_named_colors_mapping()

    color_str = ''
    for name, color in color_dict.items():
        color_str += name + ' ' * max(25-len(name), 1)
        color_str += mpt_colors.to_hex(color) + '\n'
    if out_file is None:
        print(color_str)
    else:
        with open(out_file, 'w') as f:
            f.writelines(color_str)


def single_color_val(color):
    if mpt_colors.is_color_like(color):
        return mpt_colors.to_rgba(color)
    elif isinstance(color, (list, tuple)):
        assert len(color) in [3, 4]
        for channel in color:
            assert channel >= 0 and channel < 255
        norm_color = [c / 255 for c in color]
        if len(norm_color) == 3:
            norm_color.append(1.0)
        return tuple(norm_color)
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size in [3, 4]
        return single_color_val(color.tolist())
    elif isinstance(color, (int, float)):
        assert color >= 0 and color <= 255
        return single_color_val((color, color, color, 1.0))
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')


def colors_val(colors):
    if isinstance(colors, str):
        if os.path.isfile(colors):
            with open(colors, 'r') as f:
                return [single_color_val(l.strip()) for l in f]
        else:
            return [single_color_val(s) for s in colors.split('|')]
    elif isinstance(colors, list):
        return [single_color_val(c) for c in colors]
    elif isinstance(colors, np.ndarray):
        assert colors.ndim == 2 and colors.shape[-1] in [3, 4]
        return [single_color_val(c) for c in colors]
    else:
        raise TypeError(f'Invalid type for colors: {type(colors)}')


def random_colors(num, cmap=None, rand_seed=None):
    if rand_seed is not None:
        np.random.seed(rand_seed)

    if cmap is None:
        return colors_val(np.random.random((num, 3)))
    else:
        return colors_val(cmap(np.random.random((num, ))))


def range_colors(num, cmap=None):
    colors = []
    for n in range(num):
        if cmap is None:
            color = mpt_colors.hsv_to_rgb((n / num, 1.0, 1.0))
            colors.append(single_color_val(color))
        else:
            colors.append(cmap(n / num))
    return colors
