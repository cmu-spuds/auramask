from keras.metrics import Metric, CosineSimilarity
from auramask.losses import FaceEmbedEnum
from keras_cv.layers import Resizing
import tensorflow as tf

class EmbeddingDistance(Metric):
  """Computes the loss for Adversarial Transformation Network training as described by the ReFace paper.

  In general, this loss computes the distance from computed embeddings from a set of victim models (F)

  Args:
      F ([FaceEmbedEnum]): A set of face embedding extraction models for the model to attack.
      l (float): L_{pips} loss coefficient (lambda)
  """
  def __init__(self, 
               F: set|list[FaceEmbedEnum],
               name="EmbeddingsLoss",
               **kwargs):
    super().__init__(name=name,**kwargs)
    if type(F) is set[tuple]:
      self.F_set = F
      self.F = None
    else:
      self.F = F
      self.N = len(F)
      self.F_set = FaceEmbedEnum.build_F(F)
      self.cossim = CosineSimilarity(axis=1)
    # TODO: Typechecking
    # else:
    #   raise TypeError
    
    self.distance = self.add_weight(name='norm of embeddings distance', initializer='zeros')
      
  def get_config(self):
    return {
      "name": self.name,
      "Cosine Similarity": self.cossim,
    }
        
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
    
  def update_state(self, y_true, y_pred):
    """Compute the loss across the set of target models (F)

      Args:
          y_true (_type_): Original image
          y_pred (_type_): Adversarially perturbed image

      Returns:
          tensorflow.Tensor : Normalized loss over models F
    """
    for f in self.F_set:
      try:
        model = f[0]
        aug: Resizing = f[1]
        self.distance.assign_add(self.f_cosine_similarity(aug(y_true), aug(y_pred), model))
      except Exception as e:
        print(model, aug)
        raise e
    self.distance.assign(tf.divide(self.distance, self.N))
  
  def result(self):
    return self.distance
  
  def reset_states(self):
    self.distance.assign(0)