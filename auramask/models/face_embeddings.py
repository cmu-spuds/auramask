from enum import Enum
from deepface.DeepFace import build_model
import tensorflow as tf
from keras import Sequential
from keras.layers import Subtract, Multiply
from keras_cv.layers import Resizing, Rescaling

class FaceEmbedEnum(str, Enum):
  VGGFACE = "VGG-Face"
  FACENET = "Facenet"
  FACENET512 = "Facenet512"
  OPENFACE = "OpenFace"
  DEEPFACE = "DeepFace"
  DEEPID = "DeepID"
  ARCFACE = "ArcFace"
  SFACE = "SFace"
  def get_model(self):
    d_model = build_model(model_name=self.value)
    shape = d_model.input_shape[::-1]
    if self == FaceEmbedEnum.ARCFACE or self == FaceEmbedEnum.FACENET:
      rs = Rescaling(1, offset=-1) # convert to [-1,1]
    elif self == FaceEmbedEnum.VGGFACE:
      rs = Rescaling(255, offset=0) # convert to [0, 256)
    else:
      rs = Rescaling(1, offset=0) # stay in [0, 1)
    model = Sequential([
      Resizing(shape[0], shape[1]),
      rs,
      d_model.model
    ], name=self.name)
    model.trainable = False
    for layer in model.layers:
      layer.trainable = False
      layer._name = "%s/%s"%(model.name, layer.name)
    return model
  @classmethod
  def build_F(cls, targets: list):
    F = set()
    for model_label in targets:
      assert model_label in cls
      F.add(
        model_label.get_model()
      )
    return F
  def toJSON(self):
    return self.name