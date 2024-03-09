from auramask.models.nima import NIMA
from keras.losses import Loss, mean_squared_error
import tensorflow as tf
import numpy as np

@tf.function
def _normalize_labels(labels):
  normed = labels / tf.reduce_sum(labels)
  return normed

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
    mean = self.model(y_pred)
    mean = tf.map_fn(calc_mean_score, mean)
    mean = tf.subtract(mean, 5.)    # Calculate score between [-5, 5]
    mean = tf.divide(mean, 5.)  # Convert to [-1, 1]
    return mean