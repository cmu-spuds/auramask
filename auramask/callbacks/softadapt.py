from keras import ops, backend as K
from typing import Literal
from softadapt import callbacks


class AdaptiveLossCallback(callbacks.AdaptiveLossCallback):
    def __init__(
        self,
        components: list[str],
        weights: list[float],
        frequency: int | Literal["epoch"] | Literal["batch"] = "epoch",
        beta: float = 0.1,
        accuracy_order: int = None,
        algorithm: Literal["loss-weighted"]
        | Literal["normalized"]
        | Literal["base"] = "base",
        clip_weights: bool = False,
    ):
        super().__init__(
            components, weights, frequency, beta, accuracy_order, algorithm
        )
        # self.debug = True
        self.clip_weights = clip_weights

    @property
    def weights(self):
        return self.model.loss_weights

    @weights.setter
    def weights(self, value: list):
        if self.clip_weights:
            self.model.loss_weights = [
                ops.maximum(w, K.epsilon()) for w in value
            ]  # Clip weights to avoid going below 0
        else:
            self.model.loss_weights = value
