from typing import Literal
from keras import ops, KerasTensor, Loss, backend as K
from auramask.models.nima import NIMA


def _normalize_labels(labels: KerasTensor) -> KerasTensor:
    normed = ops.divide(labels, ops.sum(labels, axis=1, keepdims=True))
    return normed


def calc_mean_score(score_dist) -> KerasTensor:
    score_dist = _normalize_labels(score_dist)
    pred_score = ops.sum(
        ops.multiply(score_dist, ops.arange(1, 11, dtype=K.floatx())), axis=1
    )
    return pred_score


class AestheticLoss(Loss):
    def __init__(
        self,
        backbone: Literal["mobilenet"]
        | Literal["nasnetmobile"]
        | Literal["inceptionresnetv2"] = "nasnetmobile",
        name="AestheticLoss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.model = NIMA(backbone=backbone)

    def get_config(self):
        return {"name": self.name, "model": self.model.name}

    def call(self, y_true: KerasTensor, y_pred: KerasTensor):
        del y_true
        mean = self.model(y_pred)
        mean = calc_mean_score(mean)
        mean = ops.subtract(1, ops.divide(mean, 10.0))  # Convert to [0, 1]
        return mean
