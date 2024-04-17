# Imports
from auramask.models.face_embeddings import FaceEmbedEnum
from keras.losses import Loss, cosine_similarity
import tensorflow as np


class EmbeddingDistanceLoss(Loss):
    """Computes the loss for Adversarial Transformation Network training as described by the ReFace paper.

    In general, this loss computes the distance from computed embeddings from a set of victim models (F)

    Args:
        F ([FaceEmbedEnum]): A set of face embedding extraction models for the model to attack.
    """

    def __init__(self, F, name="EmbeddingsLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.F = FaceEmbedEnum.build_F(F)
        self.N = np.constant(len(F), dtype=np.float32)
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
            emb_t = np.stop_gradient(f(y_true))
            emb_adv = f(y_pred)
            sim = np.negative(cosine_similarity(emb_t, emb_adv, -1))
            loss = np.add(loss, sim)
        return np.divide(loss, self.N)
