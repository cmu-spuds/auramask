from keras import metrics, backend, ops

class IQASSIMC(metrics.MeanMetricWrapper):
    def __init__(
        self,
        name="IQASSIMC",
        **kwargs
    ):
        if backend.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        from pyiqa import create_metric

        self.model = create_metric("ssimc", as_loss=False)

        def ssimc(y_true, y_pred):
            if backend.image_data_format() == "channels_last":
                y_true = ops.moveaxis(y_true, -1, 1)
                y_pred = ops.moveaxis(y_pred, -1, 1)
            diff = self.model(ref=y_true, target=y_pred)

        super().__init__(name=name, fn=ssimc, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "full_ref": True,
            "lower_better": self.model.lower_better,
            "score_range":self.model.score_range,
        }
        return {**base_config, **config}

class IQACWSSIM(metrics.MeanMetricWrapper):
    def __init__(
        self,
        name="IQACWSSIM",
        **kwargs
    ):
        if backend.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        from pyiqa import create_metric

        self.model = create_metric("cw_ssim", as_loss=False)

        def cw_ssim(y_true, y_pred):
            if backend.image_data_format() == "channels_last":
                y_true = ops.moveaxis(y_true, -1, 1)
                y_pred = ops.moveaxis(y_pred, -1, 1)
            diff = self.model(ref=y_true, target=y_pred)

        super().__init__(name=name, fn=cw_ssim, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "full_ref": True,
            "lower_better": self.model.lower_better,
            "score_range":self.model.score_range,
        }
        return {**base_config, **config}


class DSSIMObjective(metrics.MeanMetricWrapper):
    def __init__(
        self,
        k1=0.01,
        k2=0.03,
        kernel_size=3,
        max_value=1.0,
        name="DSSIMObjective",
        **kwargs
    ):
        self.kernel_size = (kernel_size, kernel_size)
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2

        def dssim(self, y_true, y_pred):
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

        super().__init__(name=name, fn=dssim, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "full_ref": True,
            "lower_better": self.model.lower_better,
            "score_range":self.model.score_range,
        }
        return {**base_config, **config}