from keras.losses import Loss
from auramask.models import LPIPS
import tensorflow as tf
from os import path

class PerceptualLoss(Loss):
  def __init__(self, 
              backbone="alex",
              spatial=False,
              l=0.2,
              model: LPIPS|None = None,
              name="PerceptualLoss",
              **kwargs):
    super().__init__(name=name,**kwargs)
    if model:
      self.model = model
    else:
      self.spatial = spatial
      self.backbone = backbone
      self.l = l
      self.model = LPIPS(backbone, spatial)
  
  def get_config(self):
    return {
      "name": self.name,
      "backbone": self.backbone,
      "spatial": self.spatial,
      "model": self.model.get_config(),
      "reduction": self.reduction,
    }
    
  def call(
    self,
    y_true, # reference_img
    y_pred, # compared_img
  ):
    return tf.multiply(self.l, self.model([y_true, y_pred]))