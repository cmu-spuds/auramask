from enum import Enum
from typing import Literal

# from deepface.DeepFace import build_model
# from deepface.modules.verification import find_threshold
from keras import layers
from keras_cv.layers import Resizing, Rescaling

from auramask.models.facenet import FaceNet
from auramask.models.vggface import VggFace


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
        input = layers.Input((None, None, 3))
        if self == FaceEmbedEnum.VGGFACE:
            x = Resizing(224, 224)(input)
            x = Rescaling(255, offset=0)(x)  # convert to [0, 256)
            model = VggFace(include_top=False, input_tensor=x)
        if self == FaceEmbedEnum.ARCFACE or self == FaceEmbedEnum.FACENET:
            x = Resizing(160, 160)(input)
            x = Rescaling(2, offset=-1)(x)  # convert to [-1,1]
            model = FaceNet(input_tensor=x)
        model.trainable = False
        print(model.summary())
        return model

        # elif self == FaceEmbedEnum.VGGFACE:
        #     lyrs.append(Rescaling(255, offset=0))  # convert to [0, 256)
        # # print(d_model.model)
        # lyrs += d_model.model.layers
        # model = Sequential(layers=lyrs, name=self.name)
        # model.trainable = False
        # for lyr in model.layers:
        #     lyr.trainable = False
        #     lyr._name = "%s/%s" % (model.name, lyr.name)
        # del d_model
        # model.build((None, 224, 224, 3))
        # print(model.summary())
        # exit()
        return model

    def get_threshold(
        self,
        distance: Literal["cosine"]
        | Literal["euclidean"]
        | Literal["euclidean_l2"] = "cosine",
    ) -> float:
        model_name = self.value
        distance_metric = distance

        base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

        thresholds = {
            # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622d
            "VGG-Face": {
                "cosine": 0.68,
                "euclidean": 1.17,
                "euclidean_l2": 1.17,
            },  # 4096d - tuned with LFW
            "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
            "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
            "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
            "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
            "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
            "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
            "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
            "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
        }

        threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

        return threshold

    @classmethod
    def build_F(cls, targets: list):
        F = set()
        for model_label in targets:
            assert model_label in cls
            F.add(model_label.get_model())
        return F

    def toJSON(self):
        return self.name
