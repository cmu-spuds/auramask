from auramask.models.nima import NIMA
from keras.losses import Loss, cosine_similarity
import tensorflow as tf
import numpy as np

@tf.function
def _normalize_labels(labels):
    return labels / tf.reduce_sum(labels)

@tf.function
def calc_mean_score(score_dist):
    score_dist = _normalize_labels(score_dist)
    return tf.reduce_sum((score_dist * tf.range(1, 11, dtype=tf.float32)))


class AestheticLoss(Loss):
  def __init__(self,
               backbone="imagenet",
               model: NIMA|None = None,
               name="AestheticLoss",
               **kwargs):
    super().__init__(name=name, **kwargs)
    if model:
      self.model = model
    else:
      self.model = NIMA(backbone=backbone, kind="aesthetic")
    
    self.model.trainable = False
    for layer in self.model.layers:
      layer.trainable = False
      layer._name = "%s/%s"%(name, layer.name)
      
  def get_config(self):
    return {
      "name": self.name,
      "model": self.model.get_config(),
      "reduction": self.reduction,
    }
  
  def call(self, y_true, y_pred):
    mean = calc_mean_score(self.model(y_pred))
    mean = tf.multiply(2., tf.divide(mean, 10.))
    mean = tf.negative(tf.subtract(mean, 1))
    return mean