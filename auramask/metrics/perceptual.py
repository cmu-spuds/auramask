from keras import metrics
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
