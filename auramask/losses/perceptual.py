from keras import Loss, ops, backend
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
        base_config = super().get_config()
        config = {"model": self.model.get_config(), "spatial": self.spatial}
        return {**base_config, **config}

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        loss = self.model([y_true, y_pred])  # in [0, 1]
        # loss = ops.subtract(loss, self.tolerance)
        # if self.tolerance > 0:
        #     loss = ops.leaky_relu(loss, negative_slope=(0.2 / self.tolerance))
        return loss


class IQAPerceptual(Loss):
    def __init__(
        self,
        tolerance: float = 0.01,
        name="lpips",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.tolerance = tolerance
        if backend.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        import pyiqa

        self.model = pyiqa.create_metric("lpips", as_loss=True)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "full_ref": True,
            "lower_better": self.model.lower_better,
            "score_range": self.model.score_range,
        }
        return {**base_config, **config}

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        # Library only supports channels first so change incoming data
        if backend.image_data_format() == "channels_last":
            y_true = ops.moveaxis(y_true, -1, 1)
            y_pred = ops.moveaxis(y_pred, -1, 1)
        return self.model(ref=y_true, target=y_pred)
