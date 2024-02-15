from keras.losses import Loss
from keras.layers import Layer
from keras.models import load_model
from keras.initializers import Zeros
import tensorflow as tf
from os import path

class WeightLayer(Layer):
    def __init__(self, weight_shape, weight_dtype, trainable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_shape = weight_shape
        self.weight_dtype = weight_dtype
        self.weight = self.add_weight('weight', shape=weight_shape, dtype=weight_dtype, trainable=trainable, initializer=Zeros()
)

    def get_config(self):
        config = super().get_config()
        config.update({
            'weight_shape': self.weight_shape,
            'weight_dtype': self.weight_dtype,
        })
        return config

    def call(self, *args, **kwargs):
        return self.weight

custom_objects = {'WeightLayer': WeightLayer}

# TODO: Host for download
_URL = 'conversion_scripts/compiled/'

class LPIPS(Loss):
  def __init__(self, 
              backbone="alex",
              spatial=False,
              l=0.2,
              name="LPIPS",
              **kwargs):
    super().__init__(name=name,**kwargs)
    self.spatial = spatial
    self.backbone = backbone
    self.l = l
    mdl_path = path.join(
        path.expanduser("~/compiled"),
        'lpips_%s%s.keras'%(backbone, 'spatial' if spatial else '')
        )
    self.net = load_model(mdl_path, custom_objects=custom_objects)
  
  def get_config(self):
    return {
      "name": self.name,
      "backbone": self.backbone,
      "spatial": self.spatial,
      "reduction": self.reduction,
    }
    
  def call(
    self,
    y_true, # reference_img
    y_pred, # compared_img
  ):
    rs_y_true = tf.image.resize(y_true, (64, 64))
    rs_y_pred = tf.image.resize(y_pred, (64, 64))
    loss = tf.multiply(self.l, self.net([rs_y_true, rs_y_pred]))
    return loss