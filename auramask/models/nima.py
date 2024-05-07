from typing import Literal
from keras.models import Model, load_model

# from keras.layers import TFSMLayer
from keras_cv.layers import Resizing
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
        self.augmenter = Resizing(224, 224)

        mdl_path = path.join(
            path.expanduser("~/compiled"), "nima_%s_%s.keras" % (kind, backbone)
        )

        self.net = load_model(mdl_path)
        # self.net = TFSMLayer(mdl_path, call_endpoint='serve')

    def get_config(self):
        return {"name": self.name, "kind": self.kind, "backbone": self.backbone}

    def call(self, x):
        x = self.augmenter(x)
        return self.net(x)
