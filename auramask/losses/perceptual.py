from keras import Loss, ops
from auramask.models.lpips import LPIPS


class PerceptualLoss(Loss):
    def __init__(
        self,
        backbone="alex",
        spatial=False,
        model: LPIPS | None = None,
        tolerance: float = 0.01,
        name="lpips",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.tolerance = tolerance
        if model:
            self.model = model
        else:
            self.model = LPIPS(backbone, spatial)

        self.model.trainable = False

    def get_config(self):
        return self.model.get_config()

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        loss = ops.squeeze(self.model([y_true, y_pred]))  # in [0, 1]
        loss = ops.subtract(loss, self.tolerance)
        if self.tolerance > 0:
            loss = ops.leaky_relu(loss, negative_slope=(0.2 / self.tolerance))
        return loss
