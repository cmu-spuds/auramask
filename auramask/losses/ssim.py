from keras.losses import Loss
from keras.src.utils.losses_utils import ReductionV2
import tensorflow as tf

class SSIMLoss(Loss):
  def __init__(self, width=224, name="SSIM", **kwargs):
    super().__init__(name=name, **kwargs)
    self.width = tf.constant(width, dtype=tf.float32)
    
  def call(self, y_true, y_pred):
    ssim = 1 - tf.image.ssim_multiscale(y_true, y_pred, max_val=self.width, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2)
    return ssim
