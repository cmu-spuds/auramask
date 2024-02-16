# from keras.metrics import Metric
# from auramask.models import LPIPS
# import tensorflow as tf

# class PerceptualSimilarity(Metric):
#   def __init__(self, 
#               backbone="alex",
#               spatial=False,
#               l=0.2,
#               model: LPIPS|None=None,
#               name="LPIPS",
#               **kwargs):
#     super().__init__(name=name,**kwargs)
#     if model:
#       self.model = model
#     else:
#       self.spatial = spatial
#       self.backbone = backbone
#       self.l = l
#       mdl_path = path.join(
#           path.expanduser("~/compiled"),
#           'lpips_%s%s.keras'%(backbone, 'spatial' if spatial else '')
#           )
#       self.net = load_model(mdl_path, custom_objects=custom_objects)
    
#     self.loss = self.add_weight(
#       shape=(), name="lpip", initializer="zeros"
#     )
    
#   def call(
#     self,
#     y_true, # reference_img
#     y_pred, # compared_img
#   ):
#     rs_y_true = tf.image.resize(y_true, (64, 64))
#     rs_y_pred = tf.image.resize(y_pred, (64, 64))
#     loss = tf.multiply(self.l, self.net([rs_y_true, rs_y_pred]))
#     return loss