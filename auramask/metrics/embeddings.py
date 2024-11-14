# Imports
from typing import Callable
from keras import metrics, KerasTensor, ops
from auramask.utils.distance import cosine_distance


class EmbeddingsDistanceMetric(metrics.Metric):
    """Computes validation metrics including true postives, true negatives, false positives, and false negatives which can be used to compute other values.

    Args:
        metrics (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self, threshold: float, d: Callable = cosine_distance, name="FV_", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.d = d
        self.threshold = threshold
        self._metrics: list[metrics.Metric] = [
            metrics.BinaryAccuracy(name="%"),
            metrics.TruePositives(name="++"),
            metrics.TrueNegatives(name="--"),
            metrics.FalseNegatives(name="+-"),
            metrics.FalsePositives(name="-+"),
        ]

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "name": self.name,
            "d": self.d.__name__,
            "threshold": self.threshold,
        }
        return {**base_config, **config}

    def update_state(
        self, y_pred: KerasTensor, y_true: KerasTensor, sample_weight=None
    ) -> KerasTensor:
        """_summary_

        Args:
            y_pred (KerasTensor): Batch of reference embeddings and source embeddings (2, B, D)
            y_true (KerasTensor): Batch of values where (1) is the same person and (0) is a different person (B,)

        Returns:
            KerasTensor: A batch-sized tensor that reflects the accuracy of the computation (1) correct prediction (0) incorrect prediction
        """
        del sample_weight
        reference, test = y_pred
        y_pred = self.d(reference, test, -1)
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
