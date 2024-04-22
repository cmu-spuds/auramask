from enum import Enum
import tensorflow as tf


@tf.function
def rgb(X):
    return X


@tf.function
def hsv_to_rgb(X):
    """Helper function to convert from hsv to rgb while passing through the gradient as
    it has not been implmented in a differentiable way.

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    return tf.grad_pass_through(tf.image.hsv_to_rgb)(X)

class ColorSpaceEnum(Enum):
    RGB = (rgb, rgb)
    HSV = (tf.image.rgb_to_hsv, hsv_to_rgb)
    YIQ = (tf.image.rgb_to_yiq, tf.image.yiq_to_rgb)
    YUV = (tf.image.rgb_to_yuv, tf.image.yuv_to_rgb)
