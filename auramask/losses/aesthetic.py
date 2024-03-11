from auramask.models.nima import NIMA
from keras.losses import Loss
# import keras.ops as np
import tensorflow as np

def _normalize_labels(labels):
  normed = labels / np.reduce_sum(labels)
  return normed

def calc_mean_score(score_dist):
  score_dist = _normalize_labels(score_dist)
  return np.reduce_sum((score_dist * np.range(1, 11, dtype=np.float32)))


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
    mean = np.map_fn(calc_mean_score, mean)
    mean = 1 - np.divide(mean, 10.)  # Convert to [0, 1]
    return mean