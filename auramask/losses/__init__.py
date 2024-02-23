from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from keras.losses import Loss
# class ModelLoss(Loss):
#   def __init__(self, model, name="ModelLoss", *args, **kwargs):
#     super().__init__(name=name, **kwargs)
#     self.model = model

#   def call(self, y_true, y_pred):
#     raise NotImplementedError

from auramask.losses.perceptual import PerceptualLoss
from auramask.losses.embeddistance import EmbeddingDistanceLoss
from auramask.losses.reface import ReFaceLoss