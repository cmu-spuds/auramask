from keras import ops, Loss


class VariationLoss(Loss):
    def __init__(
        self,
        name="VariationLoss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        del y_true

        ndims = ops.ndim(y_pred)
        if ndims == 3:
            # The input is a single image with shape [height, width, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            width_var = ops.square(y_pred[:-1, :-1, :] - y_pred[1:, :-1, :])
            height_var = ops.square(y_pred[:-1, :-1, :] - y_pred[:-1, 1:, :])
            sum_axes = (0, 1, 2)
        elif ndims == 4:
            # The input is a batch of images with shape:
            # [batch, height, width, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            width_var = ops.abs(
                ops.subtract(y_pred[:, :-1, :-1, :], y_pred[:, 1:, :-1, :])
            )
            height_var = ops.abs(
                ops.subtract(y_pred[:, :-1, :-1, :], y_pred[:, :-1, 1:, :])
            )
            sum_axes = (1, 2, 3)
        else:
            raise ValueError("'images' must be either 3 or 4-dimensional.")

        loss = ops.add(
            ops.sum(width_var, axis=sum_axes), ops.sum(height_var, axis=sum_axes)
        )

        return loss
