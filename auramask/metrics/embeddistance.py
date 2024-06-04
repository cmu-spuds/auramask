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
        if isinstance(f, FaceEmbedEnum):
            super().__init__(name=name + f.value, **kwargs)
            self.f = f.get_model()
        else:
            super().__init__(name=name + f.name, **kwargs)
            self.f = f

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {"name": self.name, "f": self.f.name}
        return {**base_config, **config}

    def update_state(
        self, y_true: KerasTensor, y_pred: KerasTensor, sample_weight=None
    ):
        emb_t = self.f(y_true, training=False)
        emb_adv = self.f(y_pred, training=False)
        return super().update_state(cosine_distance(emb_t, emb_adv, -1))


class PercentageOverThreshold(metrics.Mean):
    def __init__(self, f: FaceEmbedEnum | Model, name="PoT_", threshold=0.5, **kwargs):
        if isinstance(f, FaceEmbedEnum):
            super().__init__(name=name + f.value, **kwargs)
            self.f = f.get_model()
        else:
            super().__init__(name=name + f.name, **kwargs)
            self.f = f

        self.threshold = threshold

    def get_config(self):
        base_config = super().get_config()
        config = {"name": self.name, "f": self.f.name}
        return {**base_config, **config}

    def update_state(self, y_true, y_pred, sample_weight=None):
        emb_t = self.f(y_true, training=False)
        emb_adv = self.f(y_pred, training=False)
        dist = cosine_distance(emb_t, emb_adv, -1)
        accuracy = ops.cast(
            ops.less_equal(dist, self.threshold), dtype=backend.floatx()
        )
        return super().update_state(accuracy)
