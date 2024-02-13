# Imports
import tensorflow as tf
from tensorflow.keras.losses import Loss, cosine_similarity
import lpips_tf
from deepface.DeepFace import build_model
from deepface.commons import functions
from enum import Enum

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
               name="ReFaceLoss",
               **kwargs):
    super().__init__(name=name,**kwargs)
    self.l = l
    self.F = F
    self.N = len(F)
    self.F_set = FaceEmbedEnum.build_F(F)
    
  def get_config(self):
    return {
      "name": self.name,
      "F": self.F,
      "lambda": self.l,
      "reduction": self.reduction,
    }
  
  def call(
    self,
    y_true, # original images 
    y_pred, # perturbed images
  ):
      FLoss = self.F_loss(y_true, y_pred)
      LpipsLoss = self.lpips_loss(y_true, y_pred)
      return tf.add(FLoss, LpipsLoss, name="loss")
  
  @staticmethod
  def f_cosine_similarity(x, x_adv, f):
    """Compute the cosine distance between the embeddings of the original image and perturbed image.
    Embeddings Loss
    $$
      loss \leftarrow \dfrac{1}{\left\|\mathbb{F}\right\|} \sum^{\mathbb{F}}_{f} - \dfrac{f(x) \cdot f(x_{adv})} {\left\| f(x)\right\|_{2}\left\| f(x_{adv})\right\|_{2}}
    $$

    Args:
        x (_type_): Original image
        x_adv (_type_): Adversarially perturbed image
        f (tensorflow.keras.Model): Face embedding extraction model

    Returns:
        float: negated distance between computed embeddings
    """
    emb_t = f(x)
    emb_adv = f(x_adv)
    dist = cosine_similarity(emb_t, emb_adv, axis=1)
    dist = tf.negative(dist)
    return dist

  def F_loss(self, x, x_adv):
    """Compute the loss across the set of target models (F)

    Args:
        x (_type_): Original image
        x_adv (_type_): Adversarially perturbed image

    Returns:
        tensorflow.Tensor : Normalized loss over models F
    """
    loss = tf.Tensor(0)
    for f in self.F_set:
      model = f[0]
      in_shape = f[1]
      #TODO: convert to right shape
      loss = tf.add(loss, ReFaceLoss.f_cosine_similarity(model, x, x_adv))
    loss = tf.divide(loss, self.N)
    return loss

  def lpips_loss(self, x, x_adv):
    """Perceptual Loss ($L_{pips}$)
    $$
    loss \leftarrow loss + \lambda L_{pips}(x_{adv}, x)
    $$

    Args:
        x (_type_): Original Image
        x_adv (_type_): Adversarially Perturbed Image
        lda (float): Lambda coefficient for loss

    Returns:
        tensorflow.Tensor : Cumulative loss with L_{pips}
    """
    loss = tf.multiply(
      self.l, 
      lpips_tf.lpips(
        x_adv, 
        x, 
        model='net-lin', 
        net='alex'),
      name="Lpips Loss"
    )
    return loss

class FaceEmbedEnum(Enum):
  VGGFACE = "VGG-Face"
  FACENET = "Facenet"
  FACENET512 = "Facenet512"
  OPENFACE = "OpenFace"
  DEEPFACE = "DeepFace"
  DEEPID = "DeepID"
  ARCFACE = "ArcFace"
  SFACE = "SFace"
  def get_model(self):
    return build_model(model_name=self.value).model
  def get_target_size(self):
    return functions.find_target_size(model_name=self.value)    
  @classmethod
  def build_F(cls, targets: list):
    F = set()
    for model_label in targets:
      assert model_label in cls
      F.add(
        (
          model_label.get_model(),
          model_label.get_target_size(),
        )
      )
    return F