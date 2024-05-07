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


@tf.function
def rgb_to_yuv(X):
    yuv = tf.image.rgb_to_yuv(X)
    yuv = tf.add(yuv, [0.0, 0.5, 0.5])
    return tf.clip_by_value(yuv, 0.0, 1.0)


@tf.function
def yuv_to_rgb(X):
    yuv = tf.subtract(X, [0.0, 0.5, 0.5])
    return tf.image.yuv_to_rgb(yuv)


class ColorSpaceEnum(Enum):
    RGB = (rgb, rgb)
    HSV = (tf.image.rgb_to_hsv, hsv_to_rgb)
    YIQ = (tf.image.rgb_to_yiq, tf.image.yiq_to_rgb)
    YUV = (rgb_to_yuv, yuv_to_rgb)
