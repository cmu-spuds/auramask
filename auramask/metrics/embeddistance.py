from keras import metrics
from keras import Model, backend, ops, KerasTensor
from auramask.utils.distance import cosine_distance
from auramask.models.face_embeddings import FaceEmbedEnum


class CosineDistance(metrics.Mean):
    """Computes the distance for the given model (f) that returns a vector of embeddings with the distance metric (cosine distance by default).

    Args:
        f (FaceEmbedEnum): An instance of the FaceEmbedEnum object
    """

    def __init__(
        self,
        f: FaceEmbedEnum | Model,
        name="CosD_",
        **kwargs,
    ):
        super().__init__(name=name + f.value, **kwargs)
        self.f = f

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {"name": self.name, "f": self.f.value}
        return {**base_config, **config}

    def update_state(
        self, y_true: KerasTensor, y_pred: KerasTensor, sample_weight=None
    ):
        del sample_weight
        emb_adv = self.f.get_model()(y_pred, training=False)
        return super().update_state(cosine_distance(y_true, emb_adv, -1))


class PercentageOverThreshold(metrics.Mean):
    def __init__(self, f: FaceEmbedEnum, name="PoT_", threshold=0.5, **kwargs):
        super().__init__(name=name + f.value, **kwargs)
        self.f = f
        self.threshold = threshold

    def get_config(self):
        base_config = super().get_config()
        config = {"name": self.name, "f": self.f.name}
        return {**base_config, **config}

    def update_state(
        self, y_true: KerasTensor, y_pred: KerasTensor, sample_weight=None
    ):
        del sample_weight
        emb_adv = self.f.get_model()(y_pred, training=False)
        dist = cosine_distance(y_true, emb_adv, -1)
        accuracy = ops.cast(
            ops.less_equal(dist, self.threshold), dtype=backend.floatx()
        )
        return super().update_state(accuracy)
