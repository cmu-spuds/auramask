from keras import metrics, ops
from auramask.metrics.pyiqa import IQAMetric


def dssim(y_true, y_pred, kernel_size, c1, c2):
    patches_pred = ops.image.extract_patches(
        y_pred, kernel_size, kernel_size, padding="valid"
    )

    patches_true = ops.image.extract_patches(
        y_true, kernel_size, kernel_size, padding="valid"
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
        ops.add(ops.multiply(ops.multiply(2.0, u_true), u_pred), c1),
        ops.add(ops.multiply(2.0, covar_true_pred), c2),
    )
    denom = ops.multiply(
        ops.add(ops.add(ops.square(u_true), ops.square(u_pred)), c1),
        ops.add(ops.add(var_pred, var_true), c2),
    )
    ssim = ops.divide(ssim, denom)
    ssim = ops.mean(ops.divide(ops.subtract(1.0, ssim), 2.0), axis=[1, 2])
    return ssim


class DSSIMObjective(metrics.MeanMetricWrapper):
    def __init__(
        self,
        k1=0.01,
        k2=0.03,
        kernel_size=3,
        max_value=1.0,
        name="DSSIMObjective",
        **kwargs,
    ):
        self.config = {"k1": k1, "k2": k2, "max_value": max_value, "full_ref": True}
        super().__init__(
            name=name,
            fn=dssim,
            kernel_size=kernel_size,
            c1=(k1 * max_value) ** 2,
            c2=(k2 * max_value) ** 2,
            **kwargs,
        )

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, **self.config}


class IQASSIMC(IQAMetric):
    def __init__(self, name="IQASSIMC", **kwargs):
        super().__init__(name=name, metric_name="ssimc", **kwargs)


class IQACWSSIM(IQAMetric):
    def __init__(self, name="IQACWSSIM", **kwargs):
        super().__init__(name=name, metric_name="cw_ssim", **kwargs)
