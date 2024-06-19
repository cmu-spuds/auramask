from enum import Enum
# from keras import ops

from auramask.utils.colorspace.hsv import hsv_to_rgb, rgb_to_hsv
from auramask.utils.colorspace.yuv import rgb_to_yuv, yuv_to_rgb


def rgb(X):
    return X


class ColorSpaceEnum(Enum):
    RGB = (rgb, rgb)
    HSV = (rgb_to_hsv, hsv_to_rgb)
    YUV = (rgb_to_yuv, yuv_to_rgb)
