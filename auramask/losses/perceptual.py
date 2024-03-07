from keras.losses import Loss
from auramask.models.lpips import LPIPS
import tensorflow as tf

class PerceptualLoss(Loss):
  def __init__(self, 
              backbone="alex",
              spatial=False,
              model: LPIPS|None = None,
              name="lpips",
              **kwargs):
    super().__init__(name=name,**kwargs)
    if model:
      self.model = model
    else:
      self.model = LPIPS(backbone, spatial)

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
    
  def call(
    self,
    y_true, # reference_img
    y_pred, # compared_img
  ):
    return tf.reduce_mean(self.model([y_true, y_pred]))
