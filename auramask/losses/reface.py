# Imports
from keras.losses import Loss, CosineSimilarity
from deepface.DeepFace import build_model
from enum import Enum
import tensorflow as tf
from auramask.losses.lpips import LPIPS

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
    self.F = F
    self.N = len(F)
    self.F_set = FaceEmbedEnum.build_F(F)
    self.lpips = LPIPS(
                  backbone=backbone,
                  spatial=spatial,
                  l=l
                  )
    self.cossim = CosineSimilarity(axis=1)
    
  def get_config(self):
    dict = self.lpips.get_config()
    return {
      "name": self.name,
      "F": self.F,
      "lambda": self.l,
      "reduction": self.reduction,
      "lpips": dict,
    }
  
  def call(
    self,
    y_true, # original images 
    y_pred, # perturbed images
  ):
      FLoss = self.F_loss(y_true, y_pred)
      LpipsLoss = self.lpips(y_true, y_pred)
      return tf.add(FLoss, LpipsLoss, name="loss")
  
  def f_cosine_similarity(self, x, x_adv, f):
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
    dist = self.cossim(emb_t, emb_adv)
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
    loss = 0.0
    for f in self.F_set:
      try:
        model = f[0]
        in_shape = f[1]
        x_in = tf.image.resize(x, in_shape)
        x_adv_in = tf.image.resize(x_adv, in_shape)
        loss = tf.add(loss, self.f_cosine_similarity(x_in, x_adv_in, model))
      except Exception as e:
        print(model, in_shape)
        raise e
    loss = tf.divide(loss, self.N)
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
    model = build_model(model_name=self.value)
    shape = model.input_shape[::-1]
    model = model.model
    return (model, shape)
  @classmethod
  def build_F(cls, targets: list):
    F = set()
    for model_label in targets:
      assert model_label in cls
      F.add(
        model_label.get_model()
      )
    return F