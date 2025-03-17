# Imports
from keras import metrics, KerasTensor, ops
from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.utils.distance import (
    cosine_distance,
    euclidean_distance,
    euclidean_l2_distance,
)

def embedding_validation(y_true, y_pred, embed_net, distance_fn, threshold_value):
    emb_adv = embed_net(y_true, training=False)
    emb_true = embed_net(y_pred, training=False)
    distance = distance_fn(emb_true, emb_adv, -1)
    validated = ops.less_equal(distance, threshold_value)
    return validated

class CosineValidation(metrics.MeanMetricWrapper):
    """Computes the cosine distance of the embeddings for the given model (f) and returns a vector of truth values [0,1] demonstrating if the cosine distance is below the threshold.

    Args:

    """
    def __init__(
        self,
        f: FaceEmbedEnum,
        name="CosineValidation_",
        **kwargs,
    ):
        self.f = f

        super().__init__(
            fn=embedding_validation,
            name=name + f.value,
            embed_net=self.f.get_model(),
            distance_fn=cosine_distance,
            threshold_value=self.f.get_threshold("cosine"),
            **kwargs,
        )

        self._direction = "up"

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "name": self.name,
            "f": self.f.value,
            "threshold": self.f.get_threshold("cosine"),
        }
        return {**base_config, **config}