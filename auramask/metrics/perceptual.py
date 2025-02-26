from keras import metrics, backend, ops
from auramask.models.lpips import LPIPS


class PerceptualSimilarity(metrics.MeanMetricWrapper):
    def __init__(
        self,
        backbone="alex",
        spatial=False,
        model: LPIPS | None = None,
        name="Lpips",
        **kwargs,
    ):
        if model:
            self.model = model
        else:
            self.model = LPIPS(backbone, spatial)

        def perceptual_similarity(y_true, y_pred):
            diff = self.model([y_true, y_pred])
            return diff

        super().__init__(fn=perceptual_similarity, name=name, **kwargs)

        self._direction = "down"

    def get_config(self):
        return {
            "name": self.name,
            "model": self.model.get_config(),
        }

class IQAPerceptual(metrics.MeanMetricWrapper):
    def __init__(
        self,
        name="LPIPS_IQA",
        **kwargs
    ):
        if backend.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        from pyiqa import create_metric

        self.model = create_metric("lpips", as_loss=False)

        def perceptual_similarity(y_true, y_pred):
            if backend.image_data_format() == "channels_last":
                y_true = ops.moveaxis(y_true, -1, 1)
                y_pred = ops.moveaxis(y_pred, -1, 1)
            diff = self.model(ref=y_true, target=y_pred)

        super().__init__(name=name, fn=perceptual_similarity, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "full_ref": True,
            "lower_better": self.model.lower_better,
            "score_range":self.model.score_range,
        }
        return {**base_config, **config}