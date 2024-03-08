# Imports
from auramask.models.face_embeddings import FaceEmbedEnum
from keras.losses import Loss, CosineSimilarity
import tensorflow as tf

class EmbeddingDistanceLoss(Loss):
  """Computes the loss for Adversarial Transformation Network training as described by the ReFace paper.

  In general, this loss computes the distance from computed embeddings from a set of victim models (F)

  Args:
      F ([FaceEmbedEnum]): A set of face embedding extraction models for the model to attack.
  """
  def __init__(self, 
               F,
               name="EmbeddingsLoss",
               **kwargs):
    super().__init__(name=name,**kwargs)
    self.F = FaceEmbedEnum.build_F(F)
    self.N = tf.constant(len(F))
    self.cossim = CosineSimilarity(axis=1)
    
  def get_config(self):
    return {
      "name": self.name,
      "F": str(self.F),
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
    loss = tf.constant(0, dtype=tf.float32)
    for f in self.F_set:
      try:
        loss = tf.add(loss, self.f_cosine_similarity(y_true, y_pred, f))
      except Exception as e:
        print(f)
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
    model = f
    emb_t = model(x)
    emb_adv = model(x_adv)
    dist = self.cossim(emb_t, emb_adv)
    dist = tf.negative(dist)
    return dist