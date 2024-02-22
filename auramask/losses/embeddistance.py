# Imports
from auramask.models import FaceEmbedEnum
from keras.losses import Loss, CosineSimilarity
import tensorflow as tf
# from trackedloss import TrackedLoss
from keras.metrics import Mean
import json

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
    self.F = F
    self.N = len(F)
    self.F_set = FaceEmbedEnum.build_F(F)
    self.cossim = CosineSimilarity(axis=1)
    
    self.__step = tf.Variable(0, trainable=False, dtype=tf.int64)
    
    # tf.summary.text("F Loss Config", data=json.dumps(self.get_config()))
    
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
    loss = 0.0
    for f in self.F_set:
      try:
        loss = tf.add(loss, self.f_cosine_similarity(y_true, y_pred, f))
      except Exception as e:
        print(f)
        raise e
    loss = tf.divide(loss, self.N)
    tf.summary.scalar(name="total", data=loss, step=self.__step)
    self.__step.assign_add(1)
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
    model = f[0]
    resize = f[1]
    emb_t = model(resize(x))
    emb_adv = model(resize(x_adv))
    dist = self.cossim(emb_t, emb_adv)
    dist = tf.negative(dist)
    if self.N > 1: tf.summary.scalar(name="%s"%f[2], data=dist, step=self.__step)
    return dist

class EmbeddingDistanceLossTracked(EmbeddingDistanceLoss):
  def __init__(self,
               F,
               name="EmbeddingsLoss",
               **kwargs):
    super().__init__(name=name, F=F, **kwargs)
    self._metrics = {
      'embed_dist': Mean('embed_dist', dtype=tf.float32)
    }
    
    if self.N > 1:
      for model in F:
        self._metrics[model.name] = Mean("embed_dist_%s"%model.name.lower(), dtype=tf.float32)

    # TrackedLoss.__init__(self, metrics=metrics)
    
  def call(
    self,
    y_true,
    y_pred
  ):
    loss = super().call(y_true,y_pred)
    self._metrics['embed_dist'].update_state(loss)
    return loss
    
  def f_cosine_similarity(self, x, x_adv, f):
    loss = super().f_cosine_similarity(x, x_adv, f)
    if self.N > 1: self._metrics[f[2]].update_state(loss)
    return loss
    
  @property
  def metrics(self):
    return list(self._metrics.values())