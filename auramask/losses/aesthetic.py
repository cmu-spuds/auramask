from typing import Literal

from keras import ops, KerasTensor, Loss

from auramask.models.nima import NIMA


def _normalize_labels(labels: KerasTensor) -> KerasTensor:
    normed = labels / ops.reduce_sum(labels)
    return normed


def calc_mean_score(score_dist) -> KerasTensor:
    score_dist = _normalize_labels(score_dist)
    return ops.sum((score_dist * ops.arange(1, 11, dtype="float32")))


class AestheticLoss(Loss):
    def __init__(
        self,
        backbone: Literal["mobilenet"]
        | Literal["nasnetmobile"]
        | Literal["inceptionresnetv2"] = "mobilenet",
        kind: Literal["nima-aes"] | Literal["nima-tech"] | Literal["vila"] = "nima-aes",
        model: NIMA | None = None,
        name="AestheticLoss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if model:
            self.model = model
        else:
            if kind == "nima-aes":
                self.model = NIMA(backbone=backbone, kind="aesthetic")
                for layer in self.model.layers:
                    layer.trainable = False
                    layer._name = "%s/%s" % (name, layer.name)
            elif kind == "nima-tech":
                self.model = NIMA(backbone=backbone, kind="technical")
                for layer in self.model.layers:
                    layer.trainable = False
                    layer._name = "%s/%s" % (name, layer.name)
        self.model.trainable = False

    def get_config(self):
        return {"name": self.name, "model": self.model.name, "kind": self.model.kind}

    def call(self, y_true: KerasTensor, y_pred: KerasTensor):
        del y_true
        mean = self.model(y_pred)
        mean = ops.vectorized_map(calc_mean_score, mean)
        mean = 1 - ops.divide(mean, 10.0)  # Convert to [0, 1]
        return mean
