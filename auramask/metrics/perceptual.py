from keras.metrics import Metric
from auramask.models.lpips import LPIPS
import tensorflow as tf

class PerceptualSimilarity(Metric):
  def __init__(self, 
              backbone="alex",
              spatial=False,
              model: LPIPS|None = None,
              name="Lpips",
              **kwargs):
    super().__init__(name=name,**kwargs)
    if model:
      self.model = model
    else:
      self.model = LPIPS(backbone, spatial)
      
    self.similarity = self.add_weight(name='similarity', initializer='zeros')
      
  def get_config(self):
    return {
      "name": self.name,
      "model": self.model.get_config(),
    }
    
  def update_state(self, y_true, y_pred, sample_weight=None):
    out = tf.reduce_mean(self.model([y_true, y_pred]))
    self.similarity.assign_add(out)
  
  def result(self):
    return self.similarity
  
  def reset_states(self):
    self.similarity.assign(0)