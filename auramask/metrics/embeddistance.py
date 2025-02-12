# Imports
from typing import Callable
from keras import metrics
from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.utils.distance import cosine_distance


class FaceEmbeddingMetric(metrics.MeanMetricWrapper):
    """Computes a loss for the given model (f) that returns a vector of embeddings with the distance metric (cosine distance by default).
    This loss negates the given distance function to maximize the distance between the y_true and y_pred embeddings.

    Many embeddings have a distance threshold (d_t) which decides if the embedding reflects the same thing.
    When provided, any loss value after d_t is de-emphasized.

    Args:
        f (FaceEmbedEnum): An instance of the FaceEmbedEnum object
        d (Callable): A function with y_true and y_pred
    """

    def __init__(
        self,
        f: FaceEmbedEnum,
        d: Callable = cosine_distance,
        name="FEM_",
        **kwargs,
    ):
        self.f = f
        self.d = d
        self.net = self.f.get_model()

        def embedding_distance(y_true, y_pred):
            emb_adv = self.net(y_true, training=False)
            emb_true = self.net(y_pred, training=False)
            distance = self.d(emb_true, emb_adv, -1)
            return distance

        super().__init__(fn=embedding_distance, name=name + f.value, **kwargs)

        self._direction = "up"

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "name": self.name,
            "f": self.f.value,
            "d": self.d.__name__,
        }
        return {**base_config, **config}
