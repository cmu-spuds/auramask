from keras import Loss, ops


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

    def __call__(self, y_true, y_pred):
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


# class SSIMLoss(Loss):
#     def __init__(
#         self,
#         max_val=1.0,
#         filter_size=11,
#         filter_sigma=1.5,
#         k1=0.01,
#         k2=0.03,
#         name="SSIM",
#         **kwargs,
#     ):
#         super().__init__(name=name, **kwargs)
#         self.mv = max_val
#         self.fz = filter_size
#         self.k1 = k1
#         self.k2 = k2
#         self.filter_sigma = filter_sigma

#     def get_config(self):
#         return {
#             "name": self.name,
#             "max value": self.mv,
#             "filter size": self.fz,
#             "filter sigma": self.filter_sigma,
#             "k1": self.k1,
#             "k2": self.k2,
#             "color weights": (1.0, 1.0, 1.0),
#         }

#     def call(self, y_true, y_pred):
#         from tensorflow import image

#         loss = image.ssim(
#             y_true,
#             y_pred,
#             max_val=self.mv,
#             filter_size=self.fz,
#             filter_sigma=self.filter_sigma,
#             k1=self.k1,
#             k2=self.k2,
#         )
#         return (1 - loss) / 2.0


# # Default values obtained by Wang et al.
# _MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


# class MSSSIMLoss(SSIMLoss):
#     def __init__(self, name="MS-SSIM", power_factors=_MSSSIM_WEIGHTS, **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.power_factors = power_factors

#     def call(self, y_true, y_pred):
#         from tensorflow import image

#         loss = image.ssim_multiscale(
#             y_true,
#             y_pred,
#             max_val=self.mv,
#             filter_size=self.fz,
#             filter_sigma=self.filter_sigma,
#             k1=self.k1,
#             k2=self.k2,
#         )

#         return ops.divide(ops.subtract(1, loss), 2.0)


# class GRAYSSIM(SSIMLoss):
#     def __init__(
#         self,
#         max_val=1,
#         filter_size=11,
#         filter_sigma=1.5,
#         k1=0.01,
#         k2=0.03,
#         name="G-SSIM",
#         **kwargs,
#     ):
#         super().__init__(max_val, filter_size, filter_sigma, k1, k2, name, **kwargs)

#     def call(self, y_true, y_pred):
#         from tensorflow import image

#         y_t_gs = ops.mean(y_true, 3, keepdims=True)
#         y_p_gs = ops.mean(y_pred, 3, keepdims=True)
#         return (
#             1
#             - image.ssim(
#                 y_t_gs,
#                 y_p_gs,
#                 max_val=self.mv,
#                 filter_size=self.fz,
#                 filter_sigma=self.filter_sigma,
#                 k1=self.k1,
#                 k2=self.k2,
#             )
#         ) / 2.0


# class YUVSSIMLoss(SSIMLoss):
#     def __init__(
#         self,
#         max_val=1,
#         filter_size=11,
#         filter_sigma=1.5,
#         k1=0.01,
#         k2=0.03,
#         name="YUV-SSIM",
#         **kwargs,
#     ):
#         super().__init__(max_val, filter_size, filter_sigma, k1, k2, name, **kwargs)

#     def get_config(self):
#         tmp = super().get_config()
#         tmp["color weights"] = (0.8, 0.1, 0.1)
#         return tmp

#     def call(self, y_true, y_pred):
#         loss = ssim(
#             y_true,
#             y_pred,
#             max_val=self.mv,
#             filter_size=self.fz,
#             k1=self.k1,
#             k2=self.k2,
#             channel_weights=[
#                 0.8,
#                 0.1,
#                 0.1,
#             ],  # Weights as described in https://doi.org/10.48550/arXiv.2101.06354
#         )

#         return (1 - loss) / 2.0
