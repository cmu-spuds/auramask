# Imports
from typing import Callable
from keras import ops, Loss, KerasTensor
from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.utils.distance import cosine_distance


class FaceEmbeddingLoss(Loss):
    """Computes a loss for the given model (f) that returns a vector of embeddings with the distance metric (cosine distance by default).
    This loss negates the given distance function to maximize the distance between the y_true and y_pred embeddings.

    Many embeddings have a distance threshold (d_t) which decides if the embedding reflects the same thing.
    When provided, any loss value after d_t is de-emphasized.

    Args:
        f (FaceEmbedEnum): An instance of the FaceEmbedEnum object
        d (Callable): A function with y_true and y_pred
        d_t (float): The target of the loss optimization (default None)
    """

    def __init__(
        self,
        f: FaceEmbedEnum,
        d: Callable = cosine_distance,
        name="FE_",
        reduction="sum_over_batch_size",
        **kwargs,
    ):
        super().__init__(name=name + f.value, reduction=reduction)
        self.f = f
        self.d = d
        self.net = self.f.get_model()
        self.threshold = 1

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "name": self.name,
            "f": self.f.value,
            "d": self.d.__name__,
        }
        return {**base_config, **config}

    def call(
        self,
        y_true: KerasTensor,
        y_pred: KerasTensor,
    ) -> KerasTensor:
        emb_adv = self.net(y_pred, training=False)
        emb_true = self.net(y_true, training=False)
        distance = self.d(emb_true, emb_adv, -1)
        return ops.subtract(self.threshold, distance)


class FaceEmbeddingThresholdLoss(FaceEmbeddingLoss):
    def __init__(
        self,
        f: FaceEmbedEnum,
        threshold: float,
        d: Callable = cosine_distance,
        name="FET_",
        reduction="sum_over_batch_size",
        **kwargs,
    ):
        super().__init__(f=f, d=d, name=name, reduction=reduction, **kwargs)
        self.threshold = threshold

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {"threshold": self.threshold}
        return {**base_config, **config}

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        dist = super().call(y_true, y_pred)
        dist_thresh = ops.divide(dist, self.threshold)
        return ops.nn.leaky_relu(dist_thresh, negative_slope=0.2 * self.threshold)
