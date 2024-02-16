# Imports
from keras.losses import Loss, CosineSimilarity
from enum import Enum
from deepface.DeepFace import build_model
from keras_cv.layers import Resizing
import tensorflow as tf

class EmbeddingDistanceLoss(Loss):
  """Computes the loss for Adversarial Transformation Network training as described by the ReFace paper.

  In general, this loss computes the distance from computed embeddings from a set of victim models (F)

  Args:
      F ([FaceEmbedEnum]): A set of face embedding extraction models for the model to attack.
      l (float): L_{pips} loss coefficient (lambda)
  """
  def __init__(self, 
               F,
               name="EmbeddingsLoss",
               **kwargs):
    super().__init__(name=name,**kwargs)
    self.F = F
    self.N = len(F)
    self.F_set = FaceEmbedEnum.build_F(F)
    self.cossim = CosineSimilarity(axis=1)
    
  def get_config(self):
    return {
      "name": self.name,
      "F": self.F,
      "reduction": self.reduction,
    }
  
  def call(
    self,
    y_true, # original images 
    y_pred, # perturbed images
  ):
    """Compute the loss across the set of target models (F)

      Args:
          y_true (_type_): Original image
          y_pred (_type_): Adversarially perturbed image

      Returns:
          tensorflow.Tensor : Normalized loss over models F
    """
    loss = 0.0
    for f in self.F_set:
      try:
        model = f[0]
        aug: Resizing = f[1]
        loss = tf.add(loss, self.f_cosine_similarity(aug(y_true), aug(y_pred), model))
      except Exception as e:
        print(model, aug)
        raise e
    loss = tf.divide(loss, self.N)
    return loss      
        
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
    aug = Resizing(shape[0], shape[1])
    model = model.model
    return (model, aug)
  @classmethod
  def build_F(cls, targets: list):
    F = set()
    for model_label in targets:
      assert model_label in cls
      F.add(
        model_label.get_model()
      )
    return F