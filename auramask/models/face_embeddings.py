from enum import Enum
from deepface.DeepFace import build_model
from keras_cv.layers import Resizing

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
    return (model, aug, self.name.lower())
  @classmethod
  def build_F(cls, targets: list):
    F = set()
    for model_label in targets:
      assert model_label in cls
      F.add(
        model_label.get_model()
      )
    return F