from keras.metrics import Mean, Metric
from keras import Model
from typing import Callable
from auramask.utils.distance import cosine_distance, cosine_similarity
from auramask.models.face_embeddings import FaceEmbedEnum
import tensorflow as tf


class FaceEmbeddingDistance(Mean):
    """Computes the distance for the given model (f) that returns a vector of embeddings with the distance metric (cosine distance by default).
        
    Args:
        f (FaceEmbedEnum): An instance of the FaceEmbedEnum object
        d (Callable): A function with y_true and y_pred
    """
    def __init__(self, f: FaceEmbedEnum | Model, d: Callable = cosine_distance, name="FaceEmbeddingDistance_", **kwargs):
        if isinstance(f, FaceEmbedEnum):
            super().__init__(name=name + f.value, **kwargs)
            self.f = f.get_model()
        else:
            super().__init__(name=name + f.name, **kwargs)
            self.f = f
        self.d = d

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "name": self.name,
            "f": self.f.name,
            "d": self.d.__name__,
        }
        return {**base_config, **config}
    
    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None):
        emb_t = self.f(y_true, training=False)
        emb_adv = self.f(y_pred, training=False)
        return super().update_state(self.d(emb_t, emb_adv, -1))

class EmbeddingDistance(Metric):
    """Computes the loss for Adversarial Transformation Network training as described by the ReFace paper.

    In general, this loss computes the distance from computed embeddings from a set of victim models (F)

    Args:
        F ([FaceEmbedEnum]): A set of face embedding extraction models for the model to attack.
    """

    def __init__(
        self,
        F: list[FaceEmbedEnum],
        F_set: set = None,
        name="Embedding Distance",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.F = F
        self.F = F
        self.N = len(F)
        if F_set:
            self.F_set = F_set
        else:
            self.F_set = FaceEmbedEnum.build_F(F)

        self.count = self.add_weight("count", initializer="zeros")

        self.distance = {
            "FLoss": self.add_weight(name="FLoss", initializer="zeros"),
        }
        if self.N > 1:
            for model in F:
                self.distance[model.name.lower()] = self.add_weight(
                    name="FLoss_%s" % model.name.lower(), initializer="zeros"
                )

    def get_config(self):
        return {
            "name": self.name,
            "Cosine Similarity": self.cossim,
            "F": self.F
        }

    def f_cosine_similarity(self, x, x_adv, f):
        """Compute the cosine distance between the embeddings of the original image and perturbed image.
        Embeddings Loss
        $$
          loss \leftarrow \dfrac{1}{\left\|\mathbb{F}\right\|} \sum^{\mathbb{F}}_{f} - \dfrac{f(x) \cdot f(x_{adv})} {\left\| f(x)\right\|_{2}\left\| f(x_{adv})\right\|_{2}}
        $$

        Args:
            x (_type_): Original image
            x_adv (_type_): Adversarially perturbed image
            f (tensorflow.keras.Model): Face embedding extraction model

        Returns:
            float: negated distance between computed embeddings
        """
        model = f[0]
        aug = f[1]
        emb_t = model(aug(x))
        emb_adv = model(aug(x_adv))
        dist = self.cossim(emb_t, emb_adv)
        return tf.negative(dist)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Compute the loss across the set of target models (F)

        Args:
            y_true (_type_): Original image
            y_pred (_type_): Adversarially perturbed image

        Returns:
            tensorflow.Tensor : Normalized loss over models F
        """
        loss = 0.0
        for f in self.F_set:
            dist = self.f_cosine_similarity(y_true, y_pred, f)
            loss = tf.add(dist, loss)
            if self.N > 1:
                self.distance[f[2]].assign_add(dist)
        self.distance["FLoss"].assign_add(tf.divide(loss, self.N))
        self.count.assign_add(1)
        print(self.distance)
        print(self.count)

    def result(self):
        for key in self.distance.keys():
            self.distance[key].assign(
                tf.math.divide_no_nan(self.distance[key], self.count)
            )
        return self.distance

    def reset_states(self):
        self.count.assign(0)
        for key in self.distance.keys():
            self.distance[key].assign(0.0)
