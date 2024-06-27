from typing import Callable
from keras import Loss, layers, applications, Model

from auramask.utils import distance


class ContentLoss(Loss):
    def __init__(
        self,
        name="ContentLoss",
        content_layer: list[str] = "block5_conv2",
        distance: Callable = distance.cosine_distance,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        global model_obj

        if "model_obj" not in globals():
            model_obj = {}

        if "vgg19" not in model_obj.keys():
            inp = layers.Input(shape=(None, None, 3))
            x = applications.vgg19.preprocess_input(inp)
            model = applications.VGG19(
                weights="imagenet", include_top=False, input_tensor=x
            )
            model.trainable = False
            model_obj["vgg19"] = model

        base = model_obj["vgg19"]

        self.distance = distance

        output = None
        for layer in base.layers:
            if layer.name is content_layer:
                output = layer.output

        output = layers.Flatten()(output)

        # Feature extractor
        self.feature_extractor = Model(inputs=inp, outputs=output)
        self.feature_extractor.trainable = False

    def get_config(self):
        base_config = super().get_config()
        config = {
            "distance": self.distance.__name__,
        }
        return {**base_config, **config}

    def call(self, X, y_pred):
        y_pred = layers.Rescaling(scale=255)(y_pred)
        X = layers.Rescaling(scale=255)(X)

        X_features = self.feature_extractor(X, training=False)
        pred_features = self.feature_extractor(y_pred, training=False)

        # Add content loss
        loss = self.distance(X_features, pred_features)
        return loss
