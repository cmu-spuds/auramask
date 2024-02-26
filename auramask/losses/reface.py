# Imports
from keras.losses import Loss
import tensorflow as tf
from auramask.losses.perceptual import PerceptualLoss
from auramask.losses.embeddistance import EmbeddingDistanceLoss

class ReFaceLoss(Loss):
  """Computes the loss for Adversarial Transformation Network training as described by the ReFace paper.

  In general, this loss computes the distance from computed embeddings from a set of victim models (F)

  Args:
      F ([FaceEmbedEnum]): A set of face embedding extraction models for the model to attack.
      l (float): L_{pips} loss coefficient (lambda)
  """
  def __init__(self, 
               F,
               l=0.2,
               backbone='alex',
               spatial=False,
               name="ReFaceLoss",
               **kwargs):
    super().__init__(name=name,**kwargs)
    self.l = l
    self.lpips = PerceptualLoss(
                  backbone=backbone,
                  spatial=spatial
                  )
    self.embeddist = EmbeddingDistanceLoss(
                      F,
                      )    
  def get_config(self):
    return {
      "name": self.name,
      "reduction": self.reduction,
      "PerceptualLoss": self.lpips.get_config(),
      "EmbeddingDistanceLoss": self.embeddist.get_config(),
    }
  
  def call(
    self,
    y_true, # original images 
    y_pred, # perturbed images
  ):
    embedloss = self.embeddist(y_true, y_pred)
    perceploss = tf.multiply(self.l, self.lpips(y_true, y_pred))
    return tf.add(embedloss, perceploss)