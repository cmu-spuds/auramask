from keras.losses import Loss
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import (
    _ssim_per_channel,
    _verify_compatible_image_shapes,
)


def ssim(
    img1,
    img2,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_index_map=False,
    channel_weights=[1.0, 1.0, 1.0],
):
    # Convert to tensor if needed.
    img1 = tf.convert_to_tensor(img1, name="img1")
    img2 = tf.convert_to_tensor(img2, name="img2")
    channel_weights = tf.convert_to_tensor(channel_weights, name="c_weights")
    # Shape checking.
    _, _, checks = _verify_compatible_image_shapes(img1, img2)
    img1 = tf.identity(img1)

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = tf.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, tf.float32)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)
    ssim_per_channel, _ = _ssim_per_channel(
        img1,
        img2,
        max_val,
        filter_size,
        filter_sigma,
        k1,
        k2,
        return_index_map,
    )

    return tf.reduce_sum(ssim_per_channel * channel_weights, [-1]) / tf.reduce_sum(
        channel_weights
    )


class SSIMLoss(Loss):
    def __init__(
        self,
        max_val=1.0,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        name="SSIM",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.mv = tf.constant(max_val, dtype=tf.float32)
        self.fz = tf.constant(filter_size, dtype=tf.int32)
        self.k1 = tf.constant(k1, dtype=tf.float32)
        self.k2 = tf.constant(k2, dtype=tf.float32)
        self.filter_sigma = tf.constant(filter_sigma, dtype=tf.float32)

    def get_config(self):
        return {
            "name": self.name,
            "max value": tf.as_string(self.mv, shortest=True).numpy(),
            "filter size": tf.as_string(self.fz).numpy(),
            "filter sigma": tf.as_string(self.filter_sigma, shortest=True).numpy(),
            "k1": tf.as_string(self.k1, precision=3, scientific=True).numpy(),
            "k2": tf.as_string(self.k2, precision=3, scientific=True).numpy(),
            "color weights": (1.0, 1.0, 1.0),
        }

    def call(self, y_true, y_pred):
        loss = ssim(
            y_true,
            y_pred,
            max_val=self.mv,
            filter_size=self.fz,
            filter_sigma=self.filter_sigma,
            k1=self.k1,
            k2=self.k2,
            channel_weights=[1.0, 1.0, 1.0],
        )
        return tf.negative(loss)


# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


class MSSSIMLoss(SSIMLoss):
    def __init__(self, name="MS-SSIM", power_factors=_MSSSIM_WEIGHTS, **kwargs):
        super().__init__(name=name, **kwargs)
        self.power_factors = power_factors

    def call(self, y_true, y_pred):
        loss = tf.image.ssim_multiscale(
            y_true,
            y_pred,
            max_val=self.mv,
            filter_size=self.fz,
            filter_sigma=self.filter_sigma,
            k1=self.k1,
            k2=self.k2,
        )
        return 1 - loss


class GRAYSSIM(SSIMLoss):
    def __init__(
        self,
        max_val=1,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        name="G-SSIM",
        **kwargs,
    ):
        super().__init__(max_val, filter_size, filter_sigma, k1, k2, name, **kwargs)

    def call(self, y_true, y_pred):
        y_t_gs = tf.image.rgb_to_grayscale(y_true)
        y_p_gs = tf.image.rgb_to_grayscale(y_pred)
        return tf.negative(
            tf.image.ssim(
                y_t_gs,
                y_p_gs,
                max_val=self.mv,
                filter_size=self.fz,
                filter_sigma=self.filter_sigma,
                k1=self.k1,
                k2=self.k2,
            )
        )


class YUVSSIMLoss(SSIMLoss):
    def __init__(
        self,
        max_val=1,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        name="YUV-SSIM",
        **kwargs,
    ):
        super().__init__(max_val, filter_size, filter_sigma, k1, k2, name, **kwargs)

    def get_config(self):
        tmp = super().get_config()
        tmp["color weights"] = (0.8, 0.1, 0.1)
        return tmp

    def call(self, y_true, y_pred):
        loss = ssim(
            y_true,
            y_pred,
            max_val=self.mv,
            filter_size=self.fz,
            k1=self.k1,
            k2=self.k2,
            channel_weights=[
                0.8,
                0.1,
                0.1,
            ],  # Weights as described in https://doi.org/10.48550/arXiv.2101.06354
        )

        return 1 - loss


# class HSVSSIMLoss(SSIMLoss):
#     def get_config(self):
#         tmp = super().get_config()
#         return tmp

#     def call(self, y_true, y_pred):
#         y_true_gray = tf.image.rgb_to_grayscale
#         return super().call(y_true, y_pred)
