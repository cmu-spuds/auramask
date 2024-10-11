from typing import Callable
from keras import ops, layers, Loss, Model, applications

from auramask.utils.distance import cosine_distance
from auramask.utils.stylerefs import StyleRefs


def get_gram_matrix(feature_map, norm_by_channels=False):
    """Compute the Gram matrix of the tensor x.

    This code was adopted from @robertomest
    https://github.com/robertomest/neural-style-keras/blob/master/training.py  # NOQA

    Args:
        feature_map: A tensor of shape (batch_size, height, width, channels)
        norm_by_channels - if True, normalize the Gram Matrix by the number
        of channels.
    Returns:
        gram - a tensor representing the Gram Matrix of feature_map
    """
    feature_map = ops.cast(feature_map, "float32")
    if ops.ndim(feature_map) == 4:
        # Reshape feature map to 2D (batch_size, height * width, channels)
        B, H, W, C = ops.shape(feature_map)
        feature_map = ops.reshape(feature_map, (B, H * W, C))
        # Compute the Gram Matrix: G = F^T F
        gram = ops.matmul(ops.transpose(feature_map, axes=(0, 2, 1)), feature_map)
    else:
        raise ValueError(
            "The input tensor should be either a 3d (H, W, C) "
            "or 4d (B, H, W, C) tensor."
        )
    # Normalize the Gram matrix
    if norm_by_channels:
        denominator = ops.cast(
            ops.multiply(ops.multiply(C, H), W), "float32"
        )  # Normalization from Johnson
    else:
        denominator = ops.cast(ops.multiply(H, W), "float32")
    gram = ops.divide(gram, denominator)

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
                ops.cast(target[layer_name], "float32")
            )
        self.N = ops.convert_to_tensor(len(style_layers), "float32")

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
        y_pred = ops.multiply(y_pred, 255.0)
        pred_features = self.feature_extractor(y_pred, training=False)
        loss = ops.zeros(shape=(), dtype="float32")

        for layer_name, S in self.S.items():
            pred_layer_features = pred_features[layer_name]

            C = get_gram_matrix(pred_layer_features)

            sl = self.distance(S, C, axis=(1, 2))
            loss = ops.add(loss, ops.divide(sl, self.N))

        return loss
