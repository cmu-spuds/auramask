from keras import Model, utils, layers, backend, ops
import os

WEIGHTS_PATH = "https://huggingface.co/logasja/DeepID/resolve/main/" "model.weights.h5"


def preprocess_input(x):
    return ops.divide(x, 255.0)
    # return layers.Rescaling(scale=1.0 / 255, offset=0)(x)


def DeepID(
    weights="deepface",
    input_tensor=None,
    input_shape=(55, 47, 3),
    classes=160,
    preprocess=False,
    name="DeepID",
    image_data_format=None,
):
    if not (weights in {"deepface", None} or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `deepface` "
            "(pre-training on VGG-Face2 (serengil deepface repo)), "
            "or the path to the weights file to be loaded.  "
            f"Received: `weights={weights}.`"
        )

    if weights == "deepface" and classes != 160:
        raise ValueError(
            'If using `weights` as `"deepface"`'
            "as true, `classes` should be 160.  "
            f"Received: `classes={classes}.`"
        )

    if not image_data_format:
        image_data_format = backend.image_data_format()

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

    x = layers.Conv2D(20, (4, 4), name="Conv1", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="Pool1")(x)
    x = layers.Dropout(rate=0.99, name="D1")(x)

    x = layers.Conv2D(40, (3, 3), name="Conv2", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="Pool2")(x)
    x = layers.Dropout(rate=0.99, name="D2")(x)

    x = layers.Conv2D(60, (3, 3), name="Conv3", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="Pool3")(x)
    x = layers.Dropout(rate=0.99, name="D3")(x)

    x1 = layers.Flatten()(x)
    fc11 = layers.Dense(160, name="fc11")(x1)

    x2 = layers.Conv2D(80, (2, 2), name="Conv4", activation="relu")(x)
    x2 = layers.Flatten()(x2)
    fc12 = layers.Dense(160, name="fc12")(x2)

    y = layers.Add()([fc11, fc12])
    y = layers.Activation("relu", name="deepid")(y)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs=inputs, outputs=y, name=name)

    if weights == "deepface":
        weights_path = utils.get_file(
            "deepid.weights.h5",
            WEIGHTS_PATH,
            cache_subdir="models",
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
