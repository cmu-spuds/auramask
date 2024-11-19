from keras import Model, utils, layers, backend, ops
from keras.src.applications.imagenet_utils import (
    obtain_input_shape,
    validate_activation,
)
import os

WEIGHTS_PATH = (
    "https://huggingface.co/logasja/VGGFace2/resolve/main/" "model.weights.h5"
)

WEIGHTS_PATH_NT = (
    "https://huggingface.co/logasja/VGGFace2_NT/resolve/main/" "model.weights.h5"
)


def preprocess_input(x):
    if backend.image_data_format() == "channels_first":
        # 'RGB'->'BGR'
        if len(x.shape) == 3:
            x = ops.stack([x[i, ...] for i in (2, 1, 0)], axis=0)
        else:
            x = ops.stack([x[:, i, :] for i in (2, 1, 0)], axis=1)
    else:
        # 'RGB'->'BGR'
        x = ops.stack([x[..., i] for i in (2, 1, 0)], axis=-1)
    mean_tensor = ops.negative(
        ops.convert_to_tensor([91.4953, 103.8827, 131.0912], dtype=backend.floatx())
    )

    # Zero-center by mean pixel
    if backend.image_data_format() == "channels_first":
        mean_tensor = ops.reshape(mean_tensor, (1, 3) + (1,) * (len(x.shape) - 2))
    else:
        mean_tensor = ops.reshape(mean_tensor, (1,) * (len(x.shape) - 1) + (3,))
    x += mean_tensor

    return x


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

    # Block 1
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        64, (3, 3), activation="relu", name="vggface_block1_conv1", dtype="float32"
    )(x)
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        64, (3, 3), activation="relu", name="vggface_block1_conv2", dtype="float32"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), dtype="float32")(x)

    # Block 2
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        128, (3, 3), activation="relu", name="vggface_block2_conv1", dtype="float32"
    )(x)
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        128, (3, 3), activation="relu", name="vggface_block2_conv2", dtype="float32"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), dtype="float32")(x)

    # Block 3
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Convolution2D(
        256, (3, 3), activation="relu", name="vggface_block3_conv1", dtype="float32"
    )(x)
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        256, (3, 3), activation="relu", name="vggface_block3_conv2", dtype="float32"
    )(x)
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        256, (3, 3), activation="relu", name="vggface_block3_conv3", dtype="float32"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), dtype="float32")(x)

    # Block 4
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        512, (3, 3), activation="relu", name="vggface_block4_conv1", dtype="float32"
    )(x)
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        512, (3, 3), activation="relu", name="vggface_block4_conv2", dtype="float32"
    )(x)
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        512, (3, 3), activation="relu", name="vggface_block4_conv3", dtype="float32"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), dtype="float32")(x)

    # Block 5
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        512, (3, 3), activation="relu", name="vggface_block5_conv1", dtype="float32"
    )(x)
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        512, (3, 3), activation="relu", name="vggface_block5_conv2", dtype="float32"
    )(x)
    x = layers.ZeroPadding2D((1, 1), dtype="float32")(x)
    x = layers.Convolution2D(
        512, (3, 3), activation="relu", name="vggface_block5_conv3", dtype="float32"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), dtype="float32")(x)

    # Classification block FC
    x = layers.Convolution2D(
        4096, (7, 7), activation="relu", name="vggface_fc1", dtype="float32"
    )(x)
    x = layers.Dropout(0.5, dtype="float32")(x)
    x = layers.Convolution2D(
        4096, (1, 1), activation="relu", name="vggface_fc2", dtype="float32"
    )(x)

    if not include_top:
        x = layers.Flatten()(x)
        if pooling == "avg":
            output = layers.GlobalAveragePooling2D(dtype="float32")(x)
        elif pooling == "max":
            output = layers.GlobalMaxPooling2D(dtype="float32")(x)
        elif pooling == "l2_norm":
            output = layers.Lambda(lambda x: ops.normalize(x, axis=1), dtype="float32")(
                x
            )
    else:
        x = layers.Dropout(0.5, dtype="float32")(x)
        x = layers.Convolution2D(classes, (1, 1), name="vggface_fc3", dtype="float32")(
            x
        )
        x = layers.Flatten(dtype="float32")(x)
        validate_activation(classifier_activation, weights)
        output = layers.Activation(
            activation=classifier_activation, name="predictions", dtype="float32"
        )(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, output, name=name)

    if weights == "deepface":
        if not include_top:
            weights_path = utils.get_file(
                "vggface_notop.weights.h5",
                WEIGHTS_PATH_NT,
                cache_subdir="models",
            )
        else:
            weights_path = utils.get_file(
                "vggface.weights.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
