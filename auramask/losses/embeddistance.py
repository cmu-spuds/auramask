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
        name="FaceEmbeddingLoss_",
        reduction="sum_over_batch_size",
        **kwargs,
    ):
        super().__init__(name=name + f.value, reduction=reduction)
        self.f = f.get_model()
        self.d = d

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "name": self.name,
            "f": self.f.name,
            "d": self.d.__name__,
        }
        return {**base_config, **config}

    def call(
        self,
        y_true: KerasTensor,
        y_pred: KerasTensor,
    ) -> KerasTensor:
        emb_t = ops.stop_gradient(self.f(y_true, training=False))
        emb_adv = self.f(y_pred, training=False)
        return ops.negative(self.d(emb_t, emb_adv, -1))


class FaceEmbeddingThresholdLoss(FaceEmbeddingLoss):
    def __init__(
        self,
        f: FaceEmbedEnum,
        threshold: float,
        d: Callable = cosine_distance,
        name="FaceEmbeddingThresholdLoss",
        reduction="sum_over_batch_size",
        **kwargs,
    ):
        super().__init__(f=f, d=d, name=name, reduction=reduction, **kwargs)
        self.threshold = ops.constant(threshold, "float32")

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "threshold": ops.strings.as_string(self.threshold, precision=2).numpy()
        }
        return {**base_config, **config}

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        emb_t = ops.stop_gradient(self.f(y_true, training=False))
        emb_adv = self.f(y_pred, training=False)
        distance = self.d(emb_t, emb_adv, -1)
        dist_thresh = ops.subtract(self.threshold, distance)
        return ops.nn.leaky_relu(dist_thresh)


class EmbeddingDistanceLoss(Loss):
    """Computes the loss for Adversarial Transformation Network training as described by the ReFace paper.

    In general, this loss computes the distance from computed embeddings from a set of victim models (F)

    Args:
        F ([FaceEmbedEnum]): A set of face embedding extraction models for the model to attack.
    """

    def __init__(self, F, name="EmbeddingsLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.F = FaceEmbedEnum.build_F(F)
        self.N = ops.constant(len(F), dtype="float32")
        self.f = F

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "F": [x.value for x in self.f],
        }

    def call(
        self,
        y_true,  # original images
        y_pred,  # perturbed images
    ):
        """Compute the loss across the set of target models (F)

        Args:
            y_true (_type_): Original image
            y_pred (_type_): Adversarially perturbed image

        Returns:
            tensorflow.Tensor : Normalized loss over models F
        """
        loss = 0.0
        for f in self.F:
            emb_t = ops.stop_gradient(f(y_true))
            emb_adv = f(y_pred)
            sim = ops.negative(cosine_distance(emb_t, emb_adv, -1))
            loss = ops.add(loss, sim)
        return ops.divide(loss, self.N)
