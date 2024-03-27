from keras.losses import Loss
import tensorflow as tf

class SSIMLoss(Loss):
  def __init__(self, max_val=1.0, filter_size=11, k1=0.01, k2=0.03, name="SSIM", **kwargs):
    super().__init__(name=name, **kwargs)
    self.mv = tf.constant(max_val, dtype=tf.float32)
    self.fz = tf.constant(filter_size, dtype=tf.int32)
    self.k1 = tf.constant(k1, dtype=tf.float32)
    self.k2 = tf.constant(k2, dtype=tf.float32)
    
  def call(self, y_true, y_pred):
    ssim = 1 - tf.image.ssim_multiscale(y_true, y_pred, max_val=self.mv, filter_size=self.fz, k1=self.k1, k2=self.k2)
    return ssim
