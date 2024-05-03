from keras.models import Model, load_model
from keras.layers import Layer
from keras_cv.layers import Resizing
from keras.initializers import Zeros

# import keras.ops as np
from tensorflow import squeeze
from os import path


class WeightLayer(Layer):
    def __init__(self, weight_shape, weight_dtype, trainable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_shape = weight_shape
        self.weight_dtype = weight_dtype
        self.weight = self.add_weight(
            "weight",
            shape=weight_shape,
            dtype=weight_dtype,
            trainable=trainable,
            initializer=Zeros(),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_shape": self.weight_shape,
                "weight_dtype": self.weight_dtype,
            }
        )
        return config

    def call(self, *args, **kwargs):
        return self.weight


class LPIPS(Model):
    """Implementation of the perceptual loss model as described by "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"

    Args:
        backbone (str): Choice of "alex", "vgg", or "squeeze"
        spatial (bool): Spatial return type
    """

    def __init__(
        self, backbone="alex", spatial=False, name="PerceptualSimilarity", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.backbone = backbone
        self.spatial = spatial
        mdl_path = path.join(
            path.expanduser("~/compiled"),
            "lpips_%s%s.keras" % (backbone, "spatial" if spatial else ""),
        )
        self.augmenter = Resizing(64, 64)
        self.net = load_model(mdl_path, custom_objects={"WeightLayer": WeightLayer})

    def get_config(self):
        return {
            "name": self.name,
            "backbone": self.backbone,
        }

    def call(self, x):
        y_true, y_pred = x
        return squeeze(self.net([self.augmenter(y_true), self.augmenter(y_pred)]))
