from enum import Enum
from typing import Literal

# from deepface.modules.verification import find_threshold
from keras import layers, backend

from auramask.models.arcface import ArcFace
from auramask.models.facenet import FaceNet
from auramask.models.deepid import DeepID
from auramask.models.vggface import VggFace
# from auramask.models.openface import OpenFace
# from auramask.utils.preprocessing import rgb_to_bgr


class FaceEmbedEnum(str, Enum):
    VGGFACE = "VGG-Face"
    FACENET = "Facenet"
    FACENET512 = "Facenet512"
    # OPENFACE = "OpenFace"   # TODO: nan quickly
    # DEEPFACE = "DeepFace"  # TODO: OOM errors
    DEEPID = "DeepID"
    ARCFACE = "ArcFace"

    def get_model(self):
        global model_obj

        if "model_obj" not in globals():
            model_obj = {}

        if self.name not in model_obj.keys():
            if backend.image_data_format() == "channels_last":
                input = layers.Input((None, None, 3))
            else:
                input = layers.Input((3, None, None))
            x = layers.Rescaling(255, offset=0)(input)  # convert to [0, 255]
            # x = layers.Lambda(rgb_to_bgr)(x)

            if self == FaceEmbedEnum.VGGFACE:
                x = layers.Resizing(224, 224, name="vggface-resize")(x)
                model = VggFace(
                    include_top=False, input_tensor=x, preprocess=True, name=self.name
                )
            elif self == FaceEmbedEnum.FACENET:
                x = layers.Resizing(160, 160, name="facenet-resize")(x)
                model = FaceNet(input_tensor=x, preprocess=True, name=self.name)
            elif self == FaceEmbedEnum.FACENET512:
                x = layers.Resizing(160, 160, name="facenet-resize")(x)
                model = FaceNet(
                    input_tensor=x, classes=512, preprocess=True, name=self.name
                )
            elif self == FaceEmbedEnum.ARCFACE:
                x = layers.Resizing(112, 112, name="arcface-resize")(x)
                model = ArcFace(input_tensor=x, preprocess=True, name=self.name)
            elif self == FaceEmbedEnum.DEEPID:
                x = layers.Resizing(
                    55, 47, name="deepid-resize", pad_to_aspect_ratio=True, fill_value=0
                )(x)
                model = DeepID(input_tensor=x, preprocess=True, name=self.name)
            # elif self == FaceEmbedEnum.OPENFACE:
            #     x = layers.Resizing(96, 96, name="openface-resize")(x)
            #     model = OpenFace(input_tensor=x, preprocess=True, name=self.name)

            model.trainable = False
            model_obj[self.name] = model
        else:
            model = model_obj[self.name]

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
