from enum import Enum
from typing import Literal

# from deepface.modules.verification import find_threshold
from keras import layers

from auramask.models.arcface import ArcFace
from auramask.models.facenet import FaceNet
from auramask.models.vggface import VggFace
from auramask.utils.preprocessing import rgb_to_bgr


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
        x = layers.Rescaling(255, offset=0)(input)  # convert to [0, 255]
        x = layers.Lambda(rgb_to_bgr)(x)

        if self == FaceEmbedEnum.VGGFACE:
            x = layers.Resizing(224, 224, name="vggface-resize")(x)
            # x = applications.vgg16.preprocess_input(
            #     x
            # )
            # x = build_model(self.value).model(x)
            # model = Model(inputs=input, outputs=x, name="vggface")
            model = VggFace(include_top=False, input_tensor=x, preprocess=True)
        elif self == FaceEmbedEnum.FACENET:
            x = layers.Resizing(160, 160, name="facenet-resize")(x)
            # x = applications.imagenet_utils.preprocess_input(
            #     x, mode="tf"
            # )
            # x = build_model(self.value).model(x)
            # model = Model(inputs=input, outputs=x, name="facenet")
            model = FaceNet(input_tensor=x, preprocess=True)
        elif self == FaceEmbedEnum.FACENET512:
            x = layers.Resizing(160, 160, name="facenet-resize")(x)
            # x = applications.imagenet_utils.preprocess_input(
            #     x, mode="tf"
            # )
            # x = build_model(self.value).model(x)
            # model = Model(inputs=input, outputs=x, name="facenet512")
            model = FaceNet(input_tensor=x, classes=512, preprocess=True)
        elif self == FaceEmbedEnum.ARCFACE:
            x = layers.Resizing(112, 112, name="arcface-resize")(x)
            # x = applications.imagenet_utils.preprocess_input(
            #     x, mode="tf"
            # )
            # x = build_model(self.value).model(x)
            # model = Model(inputs=input, outputs=x, name="arcface")

            model = ArcFace(input_tensor=x, preprocess=True)

        model.trainable = False
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
