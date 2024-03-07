from auramask.models.nima import NIMA
from keras.losses import Loss, cosine_similarity
import tensorflow as tf

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
    return tf.reduce_mean(cosine_similarity(self.model(y_true), self.model(y_pred), axis=-1))