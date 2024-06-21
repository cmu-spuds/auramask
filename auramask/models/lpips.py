from typing import Literal
from keras import Model, Layer, ops, saving, utils, layers


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
            initializer="zeros",
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
        self,
        backbone: Literal["alex"] | Literal["vgg"] | Literal["squeeze"] = "alex",
        spatial=False,
        name="PerceptualSimilarity",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.backbone = backbone
        self.spatial = spatial

        mdl_path = utils.get_file(
            origin="https://github.com/cmu-spuds/lpips_conversion/releases/download/keras/lpips_%s%s.keras"
            % (backbone, "spatial" if spatial else ""),
            cache_subdir="models",
        )

        self.augmenter = layers.Resizing(64, 64)
        self.net = saving.load_model(
            mdl_path, custom_objects={"WeightLayer": WeightLayer}
        )
        self.net.trainable = False

    def get_config(self):
        return {
            "name": self.name,
            "backbone": self.backbone,
        }

    def call(self, x):
        y_true, y_pred = x
        return ops.squeeze(self.net([self.augmenter(y_true), self.augmenter(y_pred)]))
