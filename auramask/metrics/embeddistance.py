# Imports
from keras import metrics
from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.utils.distance import (
    cosine_distance,
    euclidean_distance,
    euclidean_l2_distance,
)


def embedding_distance(y_true, y_pred, embed_net, distance_fn):
    emb_adv = embed_net(y_true, training=False)
    emb_true = embed_net(y_pred, training=False)
    distance = distance_fn(emb_true, emb_adv, -1)
    return distance


class CosineDistance(metrics.MeanMetricWrapper):
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
        name="CosineDistance_",
        **kwargs,
    ):
        self.f = f

        super().__init__(
            fn=embedding_distance,
            name=name + f.value,
            embed_net=self.f.get_model(),
            distance_fn=cosine_distance,
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


class EuclideanDistance(metrics.MeanMetricWrapper):
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
        name="EuclideanDistance_",
        **kwargs,
    ):
        self.f = f

        super().__init__(
            fn=embedding_distance,
            name=name + f.value,
            embed_net=self.f.get_model(),
            distance_fn=euclidean_distance,
            **kwargs,
        )

        self._direction = "up"

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "name": self.name,
            "f": self.f.value,
            "threshold": self.f.get_threshold("euclidean"),
        }
        return {**base_config, **config}


class EuclideanL2Distance(metrics.MeanMetricWrapper):
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
        name="EuclideanL2Distance_",
        **kwargs,
    ):
        self.f = f

        super().__init__(
            fn=embedding_distance,
            name=name + f.value,
            embed_net=self.f.get_model(),
            distance_fn=euclidean_l2_distance,
            **kwargs,
        )

        self._direction = "up"

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "name": self.name,
            "f": self.f.value,
            "threshold": self.f.get_threshold("euclidean_l2"),
        }
        return {**base_config, **config}
