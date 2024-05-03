from enum import Enum
from typing import Literal
from deepface.DeepFace import build_model
from deepface.modules.verification import find_threshold
from keras import Sequential
from keras_cv.layers import Resizing, Rescaling


# TODO: Concrete function execution (https://medium.com/tinyclues-vision/optimizing-tensorflow-models-for-inference-d3636cf34034)
class FaceEmbedEnum(str, Enum):
    VGGFACE = "VGG-Face"
    FACENET = "Facenet"
    FACENET512 = "Facenet512"
    # OPENFACE = "OpenFace"  # TODO: Quickly NaN
    # DEEPFACE = "DeepFace"  # TODO: OOM errors
    DEEPID = "DeepID"
    ARCFACE = "ArcFace"

    def get_model(self):
        d_model = build_model(model_name=self.value)
        shape = d_model.input_shape[::-1]
        layers = [Resizing(shape[0], shape[1])]
        if self == FaceEmbedEnum.ARCFACE or self == FaceEmbedEnum.FACENET:
            layers.append(Rescaling(2, offset=-1))  # convert to [-1,1]
        elif self == FaceEmbedEnum.VGGFACE:
            layers.append(Rescaling(255, offset=0))  # convert to [0, 256)
        # print(d_model.model)
        layers.append(d_model.model)
        model = Sequential(layers=layers, name=self.name)
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
            layer._name = "%s/%s" % (model.name, layer.name)
        del d_model
        return model

    def get_threshold(self, distance: Literal["cosine"] | Literal["euclidean"] | Literal["euclidean_l2"] = "cosine") -> float:
        return find_threshold(model_name=self.value, distance_metric=distance)

    @classmethod
    def build_F(cls, targets: list):
        F = set()
        for model_label in targets:
            assert model_label in cls
            F.add(model_label.get_model())
        return F

    def toJSON(self):
        return self.name
