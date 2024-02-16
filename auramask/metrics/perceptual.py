from keras.metrics import Metric
from auramask.models import LPIPS
import tensorflow as tf

class PerceptualSimilarity(Metric):
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
      
    self.similarity = self.add_weight(name='similarity', initializer='zeros')
      
  def get_config(self):
    return {
      "name": self.name,
      "backbone": self.backbone,
      "spatial": self.spatial,
      "model": self.model.get_config(),
    }
    
  def update_state(self, y_true, y_pred):
    out = tf.reduce_mean(tf.multiply(self.l, self.model([y_true, y_pred])))
    self.similarity.assign(out)
  
  def result(self):
    return self.similarity
  
  def reset_states(self):
    self.similarity.assign(0)