from typing import Callable
from keras import ops, Loss, layers
from auramask.utils import distance


class VariationLoss(Loss):
    def __init__(
        self,
        name="VariationLoss",
        distance: Callable = distance.cosine_distance,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.distance = distance

    def call(self, y_true, y_pred):
        del y_true

        ndims = ops.ndim(y_pred)
        flat = layers.Flatten()
        if ndims == 3:
            # The input is a single image with shape [height, width, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            pixel_dif1 = self.distance(flat(y_pred[1:, :, :]), flat(y_pred[:-1, :, :]))
            pixel_dif2 = self.distance(flat(y_pred[:, 1:, :]), flat(y_pred[:, :-1, :]))

            # Sum for all axis. (None is an alias for all axis.)
            # sum_axis = None
        elif ndims == 4:
            # The input is a batch of images with shape:
            # [batch, height, width, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            pixel_dif1 = self.distance(
                flat(y_pred[:, 1:, :, :]), flat(y_pred[:, :-1, :, :])
            )
            pixel_dif2 = self.distance(
                flat(y_pred[:, :, 1:, :]), flat(y_pred[:, :, :-1, :])
            )

            # Only sum for the last 3 axis.
            # This results in a 1-D tensor with the total variation for each image.
            # sum_axis = [1, 2, 3]
        else:
            raise ValueError("'images' must be either 3 or 4-dimensional.")

        # Calculate the total variation by taking the absolute value of the
        # pixel-differences and summing over the appropriate axis.
        tot_var = (pixel_dif1 + pixel_dif2) / 2.0
        return tot_var
