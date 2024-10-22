# Imports
from typing import Callable
from keras import metrics, KerasTensor
from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.utils.distance import cosine_distance


class FaceValidationAccuracy(metrics.BinaryAccuracy):
    """Computes the accuracy for the given model (f) that returns a vector of embeddings with the distance metric (cosine distance by default).
    This metric counts the number of examples that fall below the threshold given by deepface and returns the mean accuracy.

    Args:
        f (FaceEmbedEnum): An instance of the FaceEmbedEnum object
        d (Callable): A function with y_true and y_pred
        d_t (float): The target of the loss optimization (default None)
    """

    def __init__(
        self,
        f: FaceEmbedEnum,
        d: Callable = cosine_distance,
        name="FaceValidation",
        **kwargs,
    ):
        super().__init__(name=name + f.value, **kwargs)
        self.f = f
        self.d = d
        self.net = self.f.get_model()

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "name": self.name,
            "f": self.f.value,
            "d": self.d.__name__,
            "threshold": self.threshold,
        }
        return {**base_config, **config}

    def update_state(
        self, X: KerasTensor, y: KerasTensor, y_true: KerasTensor, sample_weight=None
    ) -> KerasTensor:
        """_summary_

        Args:
            X (KerasTensor): Batch of adversarially perturbed (or unperturbed) in image pair
            y (KerasTensor): Batch of second in image pair (unperturbed)
            y_true (KerasTensor): Batch of values where (1) is the same person and (0) is a different person

        Returns:
            KerasTensor: A batch-sized tensor that reflects the accuracy of the computation (1) correct prediction (0) incorrect prediction
        """
        emb_adv = self.net(X, training=False)
        emb_y = self.net(y, training=False)
        y_pred = self.d(emb_y, emb_adv, -1)
        return super().update_state(y_true, y_pred, sample_weight)
