from typing import Callable
import keras
from keras import ops, layers
import enum

from auramask.utils.distance import cosine_distance


class StyleRefs(enum.Enum):
    STARRYNIGHT = "https://keras.io/img/examples/generative/neural_style_transfer/neural_style_transfer_5_1.jpg"
    ABSTRACT = "https://images.freecreatives.com/wp-content/uploads/2015/03/9a96e0bf43baa6c3b6ea7b8d3c9c7dba-d5qi6yc.jpg"


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
        features = layers.Flatten()(ops.transpose(x, (2, 0, 1)))

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
        gram = layers.Dot(axes=2)([features, features])
    else:
        raise ValueError(
            "The input tensor should be either a 3d (H, W, C) "
            "or 4d (B, H, W, C) tensor."
        )
    # Normalize the Gram matrix
    if norm_by_channels:
        denominator = C * H * W  # Normalization from Johnson
    else:
        denominator = H * W  # Normalization from Google
    gram = gram / ops.cast(denominator, x.dtype)

    if flatten:
        gram = layers.Flatten()(gram)

    return gram


def _total_variation_loss(x):
    width_var = ops.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    height_var = ops.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return ops.sum(ops.power(width_var + height_var, 1.25), axis=(1, 2))


class StyleLoss(keras.Loss):
    def __init__(
        self,
        name="StyleLoss",
        reference: StyleRefs = StyleRefs.ABSTRACT,
        distance: Callable = cosine_distance,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        inp = keras.layers.Input(shape=(None, None, 3))
        x = keras.applications.vgg19.preprocess_input(inp)
        base = keras.applications.VGG19(
            weights="imagenet", include_top=False, input_tensor=x
        )

        self.reference = reference
        self.distance = distance

        # List of layers to use for the style loss.
        self.style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]

        outputs_dict = {}
        for layer in base.layers:
            if layer.name in self.style_layer_names:
                outputs_dict[layer.name] = layer.output

        self.feature_extractor = keras.Model(inputs=inp, outputs=outputs_dict)
        self.feature_extractor.trainable = False

        style_reference_image_path = keras.utils.get_file(
            "%s.jpg" % reference.name, reference.value, cache_subdir="style"
        )
        img = keras.utils.load_img(
            style_reference_image_path, target_size=(256, 256), keep_aspect_ratio=True
        )
        img = keras.utils.img_to_array(img)
        img = keras.ops.expand_dims(img, axis=0)

        target = self.feature_extractor(img, training=False)

        self.S = {}

        for layer_name in self.style_layer_names:
            self.S[layer_name] = get_gram_matrix(
                target[layer_name], norm_by_channels=True, flatten=True
            )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "reference": self.reference.name,
            "reference_url": self.reference.value,
            "distance": self.distance.__name__,
        }
        return {**base_config, **config}

    def call(self, X, y_pred):
        del X
        y_pred = layers.Rescaling(scale=255)(y_pred)
        pred_features = self.feature_extractor(y_pred, training=False)
        loss = ops.zeros(shape=())

        for layer_name in self.style_layer_names:
            pred_layer_features = pred_features[layer_name]
            C = get_gram_matrix(
                pred_layer_features, norm_by_channels=True, flatten=True
            )
            sl = self.distance(self.S[layer_name], C)
            loss += sl / len(self.style_layer_names)

        return loss
