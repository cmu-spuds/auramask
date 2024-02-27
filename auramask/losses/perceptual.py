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

    for layer in self.model.layers:
      layer.trainable = False

    # tf.summary.text(name="Lpips Config", data=json.dumps(self.get_config()))
    
    # self.__step = tf.Variable(0, trainable=False, dtype=tf.int64)
    
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
    out = tf.reduce_mean(self.model([y_true, y_pred]))
    # tf.summary.scalar(name="lpips", data=out, step=self.__step)
    # self.__step.assign_add(1)
    return out