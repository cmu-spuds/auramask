import importlib
from typing import Literal
from keras.models import Model, load_model

# from keras.layers import TFSMLayer
import tensorflow as tf
from os import path


class NIMA(Model):
    """_summary_

    Args:
        kind (str): Choice of "aesthetic" or "technical"
        backbone: Right now only "mobilenet"
    """

    def __init__(
        self,
        kind: Literal["aesthetic"] | Literal["technical"] = "aesthetic",
        backbone: Literal["mobilenet"]
        | Literal["nasnetmobile"]
        | Literal["inceptionresnetv2"] = "mobilenet",
        name="NIMA",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.backbone = backbone
        self.kind = kind

        if backbone == "inceptionresnetv2":
            base_module = importlib.import_module(
                "keras.applications.inception_resnet_v2"
            )
        elif backbone == "mobilenet":
            base_module = importlib.import_module("keras.applications.mobilenet")
        elif backbone == "nasnetmobile":
            base_module = importlib.import_module("keras.applications.nasnet")
        else:
            raise ValueError("Provided invalid backbone option %s", backbone)

        self.pp = getattr(base_module, "preprocess_input")
        assert callable(self.pp)

        mdl_path = path.join(
            path.expanduser("~/compiled"), "nima_%s_%s.keras" % (kind, backbone)
        )

        self.net = load_model(mdl_path)
        # self.net = TFSMLayer(mdl_path, call_endpoint='serve')

    def get_config(self):
        return {"name": self.name, "kind": self.kind, "backbone": self.backbone}

    def call(self, x):
        x = tf.multiply(x, 255.0)
        # tf.print("\nConverted: ", x.shape, x.dtype, tf.reduce_min(x), tf.reduce_max(x))
        x = self.pp(x)
        # tf.print("\nPreprocessed: ", x.shape, x.dtype, tf.reduce_min(x), tf.reduce_max(x))
        return self.net(x)
