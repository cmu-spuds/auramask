# Imports
from typing import Callable
from keras import metrics, KerasTensor, ops
from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.utils.distance import cosine_distance


class FaceValidationMetrics(metrics.Metric):
    """Computes validation metrics including true postives, true negatives, false positives, and false negatives which can be used to compute other values.

    Args:
        metrics (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self, f: FaceEmbedEnum, d: Callable = cosine_distance, name="FV_", **kwargs
    ):
        super().__init__(name=name + f.value, **kwargs)
        self.f = f
        self.d = d
        self.net = self.f.get_model()
        self.threshold = self.f.get_threshold("cosine")
        self._metrics += [
            metrics.TruePositives(name="++"),
            metrics.TrueNegatives(name="--"),
            metrics.FalseNegatives(name="+-"),
            metrics.FalsePositives(name="-+"),
        ]

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
        y_pred = ops.cast(ops.less_equal(y_pred, self.threshold), "uint8")
        for metric in self._metrics:
            metric.update_state(y_true, y_pred)

    def result(self):
        results = {}
        for metric in self._metrics:
            results[metric.name] = metric.result()
        return results

    def reset_state(self):
        for metric in self._metrics:
            metric.reset_state()
