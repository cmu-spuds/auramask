from keras import Loss, ops, KerasTensor


def ssim(
    img1: KerasTensor,
    img2: KerasTensor,
    max_val: int,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_index_map: bool = False,
    channel_weights: list = [1.0, 1.0, 1.0],
):
    # Convert to tensor if needed.
    img1 = ops.convert_to_tensor(img1, name="img1")
    img2 = ops.convert_to_tensor(img2, name="img2")
    channel_weights = ops.convert_to_tensor(channel_weights, name="c_weights")

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = ops.cast(max_val, "float32")
    img1 = ops.cast(img1, "float32")
    img2 = ops.cast(img2, "float32")
    from tensorflow.python.ops.image_ops_impl import _ssim_per_channel

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

    return ops.sum(ssim_per_channel * channel_weights, [-1]) / ops.sum(channel_weights)


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
        self.mv = ops.cast(max_val, dtype="float32")
        self.fz = ops.cast(filter_size, dtype="int32")
        self.k1 = ops.cast(k1, dtype="float32")
        self.k2 = ops.cast(k2, dtype="float32")
        self.filter_sigma = ops.cast(filter_sigma, dtype="float32")

    def get_config(self):
        return {
            "name": self.name,
            "max value": self.mv,
            "filter size": self.fz,
            "filter sigma": self.filter_sigma,
            "k1": self.k1,
            "k2": self.k2,
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
        return 1 - loss


# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


class MSSSIMLoss(SSIMLoss):
    def __init__(self, name="MS-SSIM", power_factors=_MSSSIM_WEIGHTS, **kwargs):
        super().__init__(name=name, **kwargs)
        self.power_factors = power_factors

    def call(self, y_true, y_pred):
        from tensorflow import image

        loss = image.ssim_multiscale(
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
        from tensorflow import image

        y_t_gs = ops.mean(y_true, 3, keepdims=True)
        y_p_gs = ops.mean(y_pred, 3, keepdims=True)
        return 1 - image.ssim(
            y_t_gs,
            y_p_gs,
            max_val=self.mv,
            filter_size=self.fz,
            filter_sigma=self.filter_sigma,
            k1=self.k1,
            k2=self.k2,
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
