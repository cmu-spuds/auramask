from keras import Loss, ops, backend as K


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
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.kernel_size = (kernel_size, kernel_size)
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2

    def call(self, y_true, y_pred):
        patches_pred = ops.image.extract_patches(
            y_pred, self.kernel_size, self.kernel_size, padding="valid"
        )

        patches_true = ops.image.extract_patches(
            y_true, self.kernel_size, self.kernel_size, padding="valid"
        )

        # Get mean
        u_true = ops.mean(patches_true, axis=-1)
        u_pred = ops.mean(patches_pred, axis=-1)
        # Get variance
        var_true = ops.var(patches_true, axis=-1)
        var_pred = ops.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = ops.subtract(
            ops.mean(ops.multiply(patches_true, patches_pred), axis=-1),
            ops.multiply(u_true, u_pred),
        )

        ssim = ops.multiply(
            ops.add(ops.multiply(ops.multiply(2.0, u_true), u_pred), self.c1),
            ops.add(ops.multiply(2.0, covar_true_pred), self.c2),
        )
        denom = ops.multiply(
            ops.add(ops.add(ops.square(u_true), ops.square(u_pred)), self.c1),
            ops.add(ops.add(var_pred, var_true), self.c2),
        )
        ssim = ops.divide(ssim, denom)
        ssim = ops.mean(ops.divide(ops.subtract(1.0, ssim), 2.0))
        return ssim


class GRAYSSIMObjective(DSSIMObjective):
    def __init__(
        self,
        k1=0.01,
        k2=0.03,
        kernel_size=3,
        max_value=1,
        name="GSSIMObjective",
        **kwargs,
    ):
        super().__init__(k1, k2, kernel_size, max_value, name, **kwargs)

    def call(self, y_true, y_pred):
        y_t_gs = ops.image.rgb_to_grayscale(y_true)
        y_p_gs = ops.image.rgb_to_grayscale(y_pred)

        return super().call(y_t_gs, y_p_gs)


class IQASSIMC(Loss):
    def __init__(
        self,
        name="IQASSIMC",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if K.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        import pyiqa

        self.model = pyiqa.create_metric("ssimc", as_loss=True)

    def get_config(self):
        base_config = super().get_config()
        config = {"lower_better": self.model.lower_better}
        return {**base_config, **config}

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        # Library only supports channels first so change incoming data
        if K.image_data_format() == "channels_last":
            y_true = ops.moveaxis(y_true, -1, 1)
            y_pred = ops.moveaxis(y_pred, -1, 1)
        return 1 - self.model(ref=y_true, target=y_pred)


class IQACWSSIM(Loss):
    def __init__(
        self,
        name="IQACWSSIM",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if K.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        import pyiqa

        self.model = pyiqa.create_metric("cw_ssim", as_loss=True)

    def get_config(self):
        base_config = super().get_config()
        config = {"lower_better": self.model.lower_better}
        return {**base_config, **config}

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        # Library only supports channels first so change incoming data
        if K.image_data_format() == "channels_last":
            y_true = ops.moveaxis(y_true, -1, 1)
            y_pred = ops.moveaxis(y_pred, -1, 1)
        return 1 - self.model(ref=y_true, target=y_pred)
