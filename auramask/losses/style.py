from typing import Callable
from keras import ops, layers, Loss, Model, applications, backend as K

from auramask.utils.distance import cosine_distance
from auramask.utils.stylerefs import StyleRefs


def get_gram_matrix(x, norm_by_channels=False, flatten=False):
    """Compute the Gram matrix of the tensor x.

    This code was adopted from @robertomest
    https://github.com/robertomest/neural-style-keras/blob/master/training.py  # NOQA

    Args:
        x - a tensor
        norm_by_channels - if True, normalize the Gram Matrix by the number
        of channels.
    Returns:
        gram - a tensor representing the Gram Matrix of x
    """
    if ops.ndim(x) == 3:
        features = layers.Flatten(dtype="float32")(ops.transpose(x, (2, 0, 1)))

        shape = ops.shape(x)
        C, H, W = shape[0], shape[1], shape[2]

        gram = ops.dot(features, ops.transpose(features))
    elif ops.ndim(x) == 4:
        # Swap from (B, H, W, C) to (B, C, H, W)
        x = ops.transpose(x, (0, 3, 1, 2))
        shape = ops.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]

        # Reshape as a batch of 2D matrices with vectorized channels
        features = ops.reshape(x, (B, C, -1))
        # This is a batch of Gram matrices (B, C, C).
        gram = layers.Dot(axes=2, dtype="float32")([features, features])
    else:
        raise ValueError(
            "The input tensor should be either a 3d (H, W, C) "
            "or 4d (B, H, W, C) tensor."
        )
    # Normalize the Gram matrix
    if norm_by_channels:
        denominator = ops.multiply(ops.multiply(C, H), W)  # Normalization from Johnson
    else:
        denominator = ops.multiply(H, W)  # Normalization from Google
    gram = ops.divide(gram, ops.cast(denominator, "float32"))

    if flatten:
        gram = layers.Flatten(dtype="float32")(gram)

    return gram


class StyleLoss(Loss):
    def __init__(
        self,
        name="StyleLoss",
        reference: StyleRefs = StyleRefs.DIM,
        distance: Callable = cosine_distance,
        style_layers: list[str] = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ],
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.reference = reference
        self.distance = distance

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

        outputs_dict = {}
        for layer in base.layers:
            if layer.name in style_layers:
                outputs_dict[layer.name] = layer.output

        # Feature extractor
        self.feature_extractor = Model(inputs=inp, outputs=outputs_dict)
        self.feature_extractor.trainable = False

        # Style reference
        img = reference.get_img()

        target = self.feature_extractor(img, training=False)

        # Precompute gram matrix for style
        self.S = {}
        for layer_name in style_layers:
            self.S[layer_name] = get_gram_matrix(
                target[layer_name], norm_by_channels=True, flatten=True
            )
        self.N = ops.convert_to_tensor(len(style_layers), K.floatx())

    def get_config(self):
        base_config = super().get_config()
        config = {
            "reference": self.reference.name,
            "reference_url": self.reference.value["url"],
            "distance": self.distance.__name__,
        }
        return {**base_config, **config}

    def call(self, X, y_pred):
        del X
        y_pred = layers.Rescaling(scale=255, dtype="float32")(y_pred)
        pred_features = self.feature_extractor(y_pred, training=False)
        loss = ops.zeros(shape=(), dtype="float32")

        for layer_name, S in self.S.items():
            pred_layer_features = pred_features[layer_name]
            C = get_gram_matrix(
                pred_layer_features, norm_by_channels=True, flatten=True
            )
            sl = self.distance(S, C)
            loss = ops.add(loss, ops.divide(sl, self.N))
        loss = ops.cast(loss, dtype=K.floatx())
        return loss
