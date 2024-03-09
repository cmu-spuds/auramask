from keras.models import Model, load_model
from keras_cv.layers import Augmenter, Rescaling, Resizing
from os import path

class NIMA(Model):
  """_summary_

  Args:
      kind (str): Choice of "aesthetic" or "technical"
      backbone: Right now only "imagenet"
  """
  def __init__(self, 
               kind='aesthetic', 
               backbone='imagenet',
               name="NIMA",
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.backbone=backbone
    self.kind=kind
    self.augmenter = Resizing(224,224)
        
    mdl_path = path.join(
      path.expanduser("~/compiled"),
      'nima_%s_%s.keras'%(kind, backbone)
    )
    
    self.net = load_model(mdl_path)
    
  def get_config(self):
    return self().get_config()
  
  def call(
    self,
    x
  ):
    x = self.augmenter(x)
    return self.net(x)