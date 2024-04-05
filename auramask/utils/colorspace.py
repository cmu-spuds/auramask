from enum import Enum
import tensorflow as tf


class ColorSpaceEnum(Enum):
    RGB = None
    HSV = (tf.image.rgb_to_hsv, tf.image.hsv_to_rgb)
    YIQ = (tf.image.rgb_to_yiq, tf.image.yiq_to_rgb)
    YUV = (tf.image.rgb_to_yuv, tf.image.yuv_to_rgb)
