from keras import Loss, ops, KerasTensor, backend as K
from typing import Optional
# Implementations pulled from https://github.com/keras-team/keras-io/blob/master/examples/vision/zero_dce.py

"""
## Loss functions

To enable zero-reference learning in DCE-Net, we use a set of differentiable
zero-reference losses that allow us to evaluate the quality of enhanced images.
"""


class ColorConstancyLoss(Loss):
    """An implementation of the Color Constancy Loss.

    The purpose of the Color Constancy Loss is to correct the potential color deviations in the
    enhanced image and also build the relations among the three adjusted channels. It is given by

    $$L_{c o l}=\\sum_{\\forall(p, q) \\in \\varepsilon}\\left(J^p-J^q\\right)^2, \\varepsilon=\\{(R, G),(R, B),(G, B)\\}$$

    Reference:

    1. [Zero-DCE: Zero-reference Deep Curve Estimation for Low-light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
    2. [Zero-Reference Learning for Low-Light Image Enhancement (Supplementary Material)](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Guo_Zero-Reference_Deep_Curve_CVPR_2020_supplemental.pdf)
    3. [Official PyTorch implementation of Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L9)
    4. [Tensorflow implementation of Zero-DCE](https://github.com/tuvovan/Zero_DCE_TF/blob/master/src/loss.py#L10)
    5. [Keras tutorial for implementing Zero-DCE](https://keras.io/examples/vision/zero_dce/#color-constancy-loss)

    Args:
        x (KerasTensor): image.
    """

    def __init__(self, name="ColorConstancyLoss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        del y_true
        mean_rgb = ops.mean(y_pred, axis=(1, 2), keepdims=True)
        mean_red, mean_green, mean_blue = ops.split(mean_rgb, 3, axis=3)
        difference_red_green = ops.subtract(mean_red, mean_green)
        difference_red_blue = ops.subtract(mean_red, mean_blue)
        difference_green_blue = ops.subtract(mean_blue, mean_green)
        sum_of_squares = ops.sqrt(
            ops.add(
                ops.add(
                    ops.square(difference_red_green), ops.square(difference_red_blue)
                ),
                ops.square(difference_green_blue),
            )
        )
        return sum_of_squares


class ExposureControlLoss(Loss):
    """An implementation of the Exposure Constancy Loss.

    The exposure control loss measures the distance between the average intensity value of a local
    region to the well-exposedness level E which is set within [0.4, 0.7]. It is given by

    $$L_{e x p}=\\frac{1}{M} \\sum_{k=1}^M\\left|Y_k-E\\right|$$

    Reference:

    1. [Zero-DCE: Zero-reference Deep Curve Estimation for Low-light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
    2. [Zero-Reference Learning for Low-Light Image Enhancement (Supplementary Material)](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Guo_Zero-Reference_Deep_Curve_CVPR_2020_supplemental.pdf)
    3. [Official PyTorch implementation of Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74)
    4. [Tensorflow implementation of Zero-DCE](https://github.com/tuvovan/Zero_DCE_TF/blob/master/src/loss.py#L21)
    5. [Keras tutorial for implementing Zero-DCE](https://keras.io/examples/vision/zero_dce/#exposure-loss)

    Args:
        x (KerasTensor): image.
        window_size (int): The size of the window for each dimension of the input tensor for average pooling.
        mean_val (int): The average intensity value of a local region to the well-exposedness level.
    """

    def __init__(
        self, mean_val=0.6, window_size=16, name="ExposureControlLoss", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.mean_val = mean_val
        self.window_size = window_size

    def call(self, y_true, y_pred):
        """
        ### Exposure loss

        To restrain under-/over-exposed regions, we use the *exposure control loss*.
        It measures the distance between the average intensity value of a local region
        and a preset well-exposedness level (set to `0.6`).
        """
        del y_true
        x = ops.mean(y_pred, axis=-1, keepdims=True)
        mean = ops.nn.average_pool(
            x, pool_size=self.window_size, strides=self.window_size, padding="valid"
        )
        return ops.square(ops.subtract(mean, self.mean_val))


class IlluminationSmoothnessLoss(Loss):
    """An implementation of the Illumination Smoothness Loss.

    The purpose of the illumination smoothness loss is to preserve the monotonicity relations between
    neighboring pixels and it is applied to each curve parameter map.

    Reference:

    1. [Zero-DCE: Zero-reference Deep Curve Estimation for Low-light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
    2. [Zero-Reference Learning for Low-Light Image Enhancement (Supplementary Material)](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Guo_Zero-Reference_Deep_Curve_CVPR_2020_supplemental.pdf)
    3. [Official PyTorch implementation of Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L90)
    4. [Tensorflow implementation of Zero-DCE](https://github.com/tuvovan/Zero_DCE_TF/blob/master/src/loss.py#L28)
    5. [Keras tutorial for implementing Zero-DCE](https://keras.io/examples/vision/zero_dce/#illumination-smoothness-loss)

    Args:
        x (KerasTensor): image.
    """

    def __init__(self, name="IlluminationSmoothnessLoss", **kwargs):
        super().__init__(name=name, reduction="none", **kwargs)

    def call(self, y_true, y_pred):
        """
        ### Illumination smoothness loss

        To preserve the monotonicity relations between neighboring pixels, the
        *illumination smoothness loss* is added to each curve parameter map.
        """
        del y_true
        batch_size = ops.shape(y_pred)[0]
        h_x = ops.shape(y_pred)[1]
        w_x = ops.shape(y_pred)[2]
        count_h = ops.multiply(
            ops.subtract(ops.shape(y_pred)[1], 1), ops.shape(y_pred)[2]
        )
        count_w = ops.multiply(
            ops.shape(y_pred)[1], ops.subtract(ops.shape(y_pred)[2], 1)
        )
        h_tv = ops.sum(
            ops.square(ops.subtract(y_pred[:, 1:, :, :], y_pred[:, : h_x - 1, :, :]))
        )
        w_tv = ops.sum(
            ops.square(ops.subtract(y_pred[:, :, 1:, :], y_pred[:, :, : w_x - 1, :]))
        )
        batch_size = ops.cast(batch_size, dtype=K.floatx())
        count_h = ops.cast(count_h, dtype=K.floatx())
        count_w = ops.cast(count_w, dtype=K.floatx())
        return ops.mean(
            ops.multiply(
                2,
                ops.divide(
                    ops.add(ops.divide(h_tv, count_h), ops.divide(w_tv, count_w)),
                    batch_size,
                ),
            )
        )


class SpatialConsistencyLoss(Loss):
    """
    ### Spatial consistency loss

    The *spatial consistency loss* encourages spatial coherence of the enhanced image by
    preserving the contrast between neighboring regions across the input image and its enhanced version.
    """

    def __init__(self, name="SpatialConsistencyLoss", **kwargs):
        super().__init__(name=name, **kwargs)

        self.left_kernel = ops.convert_to_tensor(
            [[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=K.floatx()
        )
        self.right_kernel = ops.convert_to_tensor(
            [[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=K.floatx()
        )
        self.up_kernel = ops.convert_to_tensor(
            [[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=K.floatx()
        )
        self.down_kernel = ops.convert_to_tensor(
            [[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=K.floatx()
        )

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        original_mean = ops.mean(y_true, 3, keepdims=True)
        enhanced_mean = ops.mean(y_pred, 3, keepdims=True)
        original_pool = ops.nn.average_pool(
            original_mean, pool_size=4, strides=4, padding="VALID"
        )
        enhanced_pool = ops.nn.average_pool(
            enhanced_mean, pool_size=4, strides=4, padding="VALID"
        )

        d_original_left = ops.nn.conv(
            original_pool,
            self.left_kernel,
            strides=1,
            padding="SAME",
        )
        d_original_right = ops.nn.conv(
            original_pool,
            self.right_kernel,
            strides=1,
            padding="SAME",
        )
        d_original_up = ops.nn.conv(
            original_pool, self.up_kernel, strides=1, padding="SAME"
        )
        d_original_down = ops.nn.conv(
            original_pool,
            self.down_kernel,
            strides=1,
            padding="SAME",
        )

        d_enhanced_left = ops.nn.conv(
            enhanced_pool,
            self.left_kernel,
            strides=1,
            padding="SAME",
        )
        d_enhanced_right = ops.nn.conv(
            enhanced_pool,
            self.right_kernel,
            strides=1,
            padding="SAME",
        )
        d_enhanced_up = ops.nn.conv(
            enhanced_pool, self.up_kernel, strides=1, padding="SAME"
        )
        d_enhanced_down = ops.nn.conv(
            enhanced_pool,
            self.down_kernel,
            strides=1,
            padding="SAME",
        )

        d_left = ops.square(ops.subtract(d_original_left, d_enhanced_left))
        d_right = ops.square(ops.subtract(d_original_right, d_enhanced_right))
        d_up = ops.square(ops.subtract(d_original_up, d_enhanced_up))
        d_down = ops.square(ops.subtract(d_original_down, d_enhanced_down))
        return ops.add(ops.add(d_left, d_right), ops.add(d_up, d_down))


class ZeroDCEEnsemble(Loss):
    def __init__(
        self, weights: Optional[list] = None, name="ZeroDCE Ensemble", **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.losses: list[Loss] = [
            ColorConstancyLoss(),
            IlluminationSmoothnessLoss(),
            ExposureControlLoss(),
            SpatialConsistencyLoss(),
        ]

        if not weights:
            self.weights: list[float] = [2.5, 100.0, 5.0, 1.0]
        else:
            self.weights = weights

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        total_loss = 0.0
        for i, lss in enumerate(self.losses):
            tmp_loss = ops.multiply(self.weights[i], lss(y_true, y_pred))
            total_loss = ops.add(total_loss, tmp_loss)
        return total_loss

    def get_config(self):
        base_config = super().get_config()
        config = {
            k.name: k.get_config | {"weight": w}
            for k, w in zip(self.losses, self.weights)
        }
        return {**base_config, **config}
