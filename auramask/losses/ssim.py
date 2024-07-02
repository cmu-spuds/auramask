from keras import Loss, ops, KerasTensor, backend as K


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
    img1 = ops.convert_to_tensor(img1)
    img2 = ops.convert_to_tensor(img2)
    channel_weights = ops.convert_to_tensor(channel_weights)

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


class DSSIMObjective(Loss):
    """Difference of Structural Similarity (DSSIM loss function).
    Clipped between 0 and 0.5

    Note : You should add a regularization term like a l2 loss in addition to this one.
    Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
           not be the `kernel_size` for an output of 32.

    # Arguments
        k1: Parameter of the SSIM (default 0.01)
        k2: Parameter of the SSIM (default 0.03)
        kernel_size: Size of the sliding window (default 3)
        max_value: Max value of the output (default 1.0)
    """

    def __init__(
        self,
        k1=0.01,
        k2=0.03,
        kernel_size=3,
        max_value=1.0,
        name="DSSIMObjective",
        reduction="sum_over_batch_size",
        **kwargs,
    ):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()

    def extract_image_patches(
        self, x, ksizes, ssizes, padding="VALID", data_format="channels_last"
    ):
        kernel = [1, ksizes[0], ksizes[1], 1]
        strides = [1, ssizes[0], ssizes[1], 1]
        if data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))
        bs_i, w_i, h_i, ch_i = ops.shape(x)
        patches = ops.image.extract_patches(x, kernel, strides, 1, padding=padding)
        bs, w, h, ch = ops.shape(patches)
        patches = ops.reshape(
            ops.transpose(
                ops.reshape(patches, [-1, w, h, ops.floor_divide(ch, ch_i), ch_i]),
                [0, 1, 2, 4, 3],
            ),
            [-1, w, h, ch_i, ksizes[0], ksizes[1]],
        )
        if data_format == "channels_last":
            patches = ops.transpose(patches, [0, 1, 2, 4, 5, 3])
        return patches

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a
        # gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [self.kernel_size, self.kernel_size]
        y_true = ops.reshape(y_true, [-1] + list(ops.shape(y_pred)[1:]))
        y_pred = ops.reshape(y_pred, [-1] + list(ops.shape(y_pred)[1:]))

        patches_pred = self.extract_image_patches(
            y_pred, kernel, kernel, padding="valid", data_format=self.dim_ordering
        )

        patches_true = self.extract_image_patches(
            y_true, kernel, kernel, padding="valid", data_format=self.dim_ordering
        )

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = ops.shape(patches_pred)
        patches_pred = ops.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = ops.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = ops.mean(patches_true, axis=-1)
        u_pred = ops.mean(patches_pred, axis=-1)
        # Get variance
        var_true = ops.var(patches_true, axis=-1)
        var_pred = ops.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = (
            ops.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred
        )

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (ops.square(u_true) + ops.square(u_pred) + self.c1) * (
            var_pred + var_true + self.c2
        )
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return (1.0 - ssim) / 2.0


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
        self.mv = max_val
        self.fz = filter_size
        self.k1 = k1
        self.k2 = k2
        self.filter_sigma = filter_sigma

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
        from tensorflow import image

        loss = image.ssim(
            y_true,
            y_pred,
            max_val=self.mv,
            filter_size=self.fz,
            filter_sigma=self.filter_sigma,
            k1=self.k1,
            k2=self.k2,
        )
        return (1 - loss) / 2.0


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

        print(loss)

        return (1 - loss) / 2.0


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
        return (
            1
            - image.ssim(
                y_t_gs,
                y_p_gs,
                max_val=self.mv,
                filter_size=self.fz,
                filter_sigma=self.filter_sigma,
                k1=self.k1,
                k2=self.k2,
            )
        ) / 2.0


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

        return (1 - loss) / 2.0


# class HSVSSIMLoss(SSIMLoss):
#     def get_config(self):
#         tmp = super().get_config()
#         return tmp

#     def call(self, y_true, y_pred):
#         y_true_gray = tf.image.rgb_to_grayscale
#         return super().call(y_true, y_pred)
