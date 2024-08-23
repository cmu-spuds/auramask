from keras import Model, utils, layers, backend, ops
from keras.src.applications.imagenet_utils import (
    obtain_input_shape,
    validate_activation,
)
import os

WEIGHTS_PATH = (
    "https://github.com/serengil/deepface_models/releases/download/"
    "v1.0/vgg_face_weights.h5"
)


def preprocess_input(x):
    if backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1
    b, g, r = ops.split(x, 3, axis=axis)
    b = ops.subtract(b, 93.540)
    g = ops.subtract(g, 104.7624)
    r = ops.subtract(r, 129.1863)
    return ops.concatenate([b, g, r], axis=axis)


def VggFace(
    include_top=True,
    weights="deepface",
    input_tensor=None,
    input_shape=None,
    pooling="l2_norm",
    classes=2622,
    classifier_activation="softmax",
    preprocess=False,
    name="VggFace",
):
    if not (weights in {"deepface", None} or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `deepface` "
            "(pre-training on VGG-Face2 (serengil deepface repo)), "
            "or the path to the weights file to be loaded.  "
            f"Received: `weights={weights}.`"
        )

    if weights == "deepface" and include_top and classes != 2622:
        raise ValueError(
            'If using `weights` as `"deepface"` with `include_top` '
            "as true, `classes` should be 2622.  "
            f"Received: `classes={classes}.`"
        )

    # Determine proper input shape
    input_shape = obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights="imagenet" if weights == "deepface" else weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if preprocess:
        x = preprocess_input(img_input)
    else:
        x = img_input

    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(64, (3, 3), activation="relu")(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(128, (3, 3), activation="relu")(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(256, (3, 3), activation="relu")(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(256, (3, 3), activation="relu")(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(256, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(512, (3, 3), activation="relu")(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(512, (3, 3), activation="relu")(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(512, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(512, (3, 3), activation="relu")(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(512, (3, 3), activation="relu")(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(512, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classification block
    x = layers.Convolution2D(4096, (7, 7), activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Convolution2D(4096, (1, 1), activation="relu")(x)

    if not include_top:
        output = x

    x = layers.Dropout(0.5)(x)
    x = layers.Convolution2D(classes, (1, 1))(x)
    x = layers.Flatten()(x)
    validate_activation(classifier_activation, weights)
    x = layers.Activation(activation=classifier_activation, name="predictions")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if weights == "deepface":
        weights_path = utils.get_file(
            "vggface.h5",
            WEIGHTS_PATH,
            cache_subdir="models",
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if not include_top:
        output = layers.Flatten()(output)
        if pooling == "avg":
            output = layers.GlobalAveragePooling2D()(output)
        elif pooling == "max":
            output = layers.GlobalMaxPooling2D()(output)
        elif pooling == "l2_norm":
            output = layers.UnitNormalization()(output)

        model = Model(inputs, output, name=name)

    return model
