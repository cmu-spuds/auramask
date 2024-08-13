from keras import Model, utils, layers, backend, ops

if backend.backend() == "tensorflow":
    from tensorflow import nn

    lrn = nn.lrn
elif backend.backend() == "torch":
    from torch import nn

    lrn = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75)
import os

WEIGHTS_PATH = (
    "https://github.com/serengil/deepface_models/releases/download/"
    "v1.0/openface_weights.h5"
)


def preprocess_input(x):
    return layers.Rescaling(scale=1.0 / 255, offset=0)(x)


def OpenFace(
    weights="deepface",
    input_tensor=None,
    input_shape=(96, 96, 3),
    classes=128,
    preprocess=False,
    name="DeepID",
):
    if not (weights in {"deepface", None} or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `deepface` "
            "(pre-training on VGG-Face2 (serengil deepface repo)), "
            "or the path to the weights file to be loaded.  "
            f"Received: `weights={weights}.`"
        )

    if weights == "deepface" and classes != 128:
        raise ValueError(
            'If using `weights` as `"deepface"`'
            "as true, `classes` should be 128.  "
            f"Received: `classes={classes}.`"
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

    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=0.00001, name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Lambda(lambda x: lrn(x, alpha=1e-4, beta=0.75), name="lrn_1")(x)
    x = layers.Conv2D(64, (1, 1), name="conv2")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=0.00001, name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(192, (3, 3), name="conv3")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=0.00001, name="bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.Lambda(lambda x: lrn(x, alpha=1e-4, beta=0.75), name="lrn_2")(
        x
    )  # x is equal added
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Inception3a
    inception_3a_3x3 = layers.Conv2D(96, (1, 1), name="inception_3a_3x3_conv1")(x)
    inception_3a_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3a_3x3_bn1"
    )(inception_3a_3x3)
    inception_3a_3x3 = layers.Activation("relu")(inception_3a_3x3)
    inception_3a_3x3 = layers.ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = layers.Conv2D(128, (3, 3), name="inception_3a_3x3_conv2")(
        inception_3a_3x3
    )
    inception_3a_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3a_3x3_bn2"
    )(inception_3a_3x3)
    inception_3a_3x3 = layers.Activation("relu")(inception_3a_3x3)

    inception_3a_5x5 = layers.Conv2D(16, (1, 1), name="inception_3a_5x5_conv1")(x)
    inception_3a_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3a_5x5_bn1"
    )(inception_3a_5x5)
    inception_3a_5x5 = layers.Activation("relu")(inception_3a_5x5)
    inception_3a_5x5 = layers.ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = layers.Conv2D(32, (5, 5), name="inception_3a_5x5_conv2")(
        inception_3a_5x5
    )
    inception_3a_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3a_5x5_bn2"
    )(inception_3a_5x5)
    inception_3a_5x5 = layers.Activation("relu")(inception_3a_5x5)

    inception_3a_pool = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = layers.Conv2D(32, (1, 1), name="inception_3a_pool_conv")(
        inception_3a_pool
    )
    inception_3a_pool = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3a_pool_bn"
    )(inception_3a_pool)
    inception_3a_pool = layers.Activation("relu")(inception_3a_pool)
    inception_3a_pool = layers.ZeroPadding2D(padding=((3, 4), (3, 4)))(
        inception_3a_pool
    )

    inception_3a_1x1 = layers.Conv2D(64, (1, 1), name="inception_3a_1x1_conv")(x)
    inception_3a_1x1 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3a_1x1_bn"
    )(inception_3a_1x1)
    inception_3a_1x1 = layers.Activation("relu")(inception_3a_1x1)

    inception_3a = ops.concatenate(
        [inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1],
        axis=-1,
    )

    # Inception3b
    inception_3b_3x3 = layers.Conv2D(96, (1, 1), name="inception_3b_3x3_conv1")(
        inception_3a
    )
    inception_3b_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3b_3x3_bn1"
    )(inception_3b_3x3)
    inception_3b_3x3 = layers.Activation("relu")(inception_3b_3x3)
    inception_3b_3x3 = layers.ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = layers.Conv2D(128, (3, 3), name="inception_3b_3x3_conv2")(
        inception_3b_3x3
    )
    inception_3b_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3b_3x3_bn2"
    )(inception_3b_3x3)
    inception_3b_3x3 = layers.Activation("relu")(inception_3b_3x3)

    inception_3b_5x5 = layers.Conv2D(32, (1, 1), name="inception_3b_5x5_conv1")(
        inception_3a
    )
    inception_3b_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3b_5x5_bn1"
    )(inception_3b_5x5)
    inception_3b_5x5 = layers.Activation("relu")(inception_3b_5x5)
    inception_3b_5x5 = layers.ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = layers.Conv2D(64, (5, 5), name="inception_3b_5x5_conv2")(
        inception_3b_5x5
    )
    inception_3b_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3b_5x5_bn2"
    )(inception_3b_5x5)
    inception_3b_5x5 = layers.Activation("relu")(inception_3b_5x5)

    inception_3b_pool = layers.Lambda(lambda x: x**2, name="power2_3b")(inception_3a)
    inception_3b_pool = layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(
        inception_3b_pool
    )
    inception_3b_pool = layers.Lambda(lambda x: x * 9, name="mult9_3b")(
        inception_3b_pool
    )
    inception_3b_pool = layers.Lambda(lambda x: ops.sqrt(x), name="sqrt_3b")(
        inception_3b_pool
    )
    inception_3b_pool = layers.Conv2D(64, (1, 1), name="inception_3b_pool_conv")(
        inception_3b_pool
    )
    inception_3b_pool = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3b_pool_bn"
    )(inception_3b_pool)
    inception_3b_pool = layers.Activation("relu")(inception_3b_pool)
    inception_3b_pool = layers.ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = layers.Conv2D(64, (1, 1), name="inception_3b_1x1_conv")(
        inception_3a
    )
    inception_3b_1x1 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3b_1x1_bn"
    )(inception_3b_1x1)
    inception_3b_1x1 = layers.Activation("relu")(inception_3b_1x1)

    inception_3b = ops.concatenate(
        [inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1],
        axis=-1,
    )

    # Inception3c
    inception_3c_3x3 = layers.Conv2D(
        128, (1, 1), strides=(1, 1), name="inception_3c_3x3_conv1"
    )(inception_3b)
    inception_3c_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3c_3x3_bn1"
    )(inception_3c_3x3)
    inception_3c_3x3 = layers.Activation("relu")(inception_3c_3x3)
    inception_3c_3x3 = layers.ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
    inception_3c_3x3 = layers.Conv2D(
        256, (3, 3), strides=(2, 2), name="inception_3c_3x3_conv" + "2"
    )(inception_3c_3x3)
    inception_3c_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3c_3x3_bn" + "2"
    )(inception_3c_3x3)
    inception_3c_3x3 = layers.Activation("relu")(inception_3c_3x3)

    inception_3c_5x5 = layers.Conv2D(
        32, (1, 1), strides=(1, 1), name="inception_3c_5x5_conv1"
    )(inception_3b)
    inception_3c_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3c_5x5_bn1"
    )(inception_3c_5x5)
    inception_3c_5x5 = layers.Activation("relu")(inception_3c_5x5)
    inception_3c_5x5 = layers.ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
    inception_3c_5x5 = layers.Conv2D(
        64, (5, 5), strides=(2, 2), name="inception_3c_5x5_conv" + "2"
    )(inception_3c_5x5)
    inception_3c_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_3c_5x5_bn" + "2"
    )(inception_3c_5x5)
    inception_3c_5x5 = layers.Activation("relu")(inception_3c_5x5)

    inception_3c_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(
        inception_3c_pool
    )

    inception_3c = ops.concatenate(
        [inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=-1
    )

    # inception 4a
    inception_4a_3x3 = layers.Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_4a_3x3_conv" + "1"
    )(inception_3c)
    inception_4a_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4a_3x3_bn" + "1"
    )(inception_4a_3x3)
    inception_4a_3x3 = layers.Activation("relu")(inception_4a_3x3)
    inception_4a_3x3 = layers.ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
    inception_4a_3x3 = layers.Conv2D(
        192, (3, 3), strides=(1, 1), name="inception_4a_3x3_conv" + "2"
    )(inception_4a_3x3)
    inception_4a_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4a_3x3_bn" + "2"
    )(inception_4a_3x3)
    inception_4a_3x3 = layers.Activation("relu")(inception_4a_3x3)

    inception_4a_5x5 = layers.Conv2D(
        32, (1, 1), strides=(1, 1), name="inception_4a_5x5_conv1"
    )(inception_3c)
    inception_4a_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4a_5x5_bn1"
    )(inception_4a_5x5)
    inception_4a_5x5 = layers.Activation("relu")(inception_4a_5x5)
    inception_4a_5x5 = layers.ZeroPadding2D(padding=(2, 2))(inception_4a_5x5)
    inception_4a_5x5 = layers.Conv2D(
        64, (5, 5), strides=(1, 1), name="inception_4a_5x5_conv" + "2"
    )(inception_4a_5x5)
    inception_4a_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4a_5x5_bn" + "2"
    )(inception_4a_5x5)
    inception_4a_5x5 = layers.Activation("relu")(inception_4a_5x5)

    inception_4a_pool = layers.Lambda(lambda x: x**2, name="power2_4a")(inception_3c)
    inception_4a_pool = layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(
        inception_4a_pool
    )
    inception_4a_pool = layers.Lambda(lambda x: x * 9, name="mult9_4a")(
        inception_4a_pool
    )
    inception_4a_pool = layers.Lambda(lambda x: ops.sqrt(x), name="sqrt_4a")(
        inception_4a_pool
    )

    inception_4a_pool = layers.Conv2D(
        128, (1, 1), strides=(1, 1), name="inception_4a_pool_conv" + ""
    )(inception_4a_pool)
    inception_4a_pool = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4a_pool_bn" + ""
    )(inception_4a_pool)
    inception_4a_pool = layers.Activation("relu")(inception_4a_pool)
    inception_4a_pool = layers.ZeroPadding2D(padding=(2, 2))(inception_4a_pool)

    inception_4a_1x1 = layers.Conv2D(
        256, (1, 1), strides=(1, 1), name="inception_4a_1x1_conv" + ""
    )(inception_3c)
    inception_4a_1x1 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4a_1x1_bn" + ""
    )(inception_4a_1x1)
    inception_4a_1x1 = layers.Activation("relu")(inception_4a_1x1)

    inception_4a = ops.concatenate(
        [inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1],
        axis=-1,
    )

    # inception4e
    inception_4e_3x3 = layers.Conv2D(
        160, (1, 1), strides=(1, 1), name="inception_4e_3x3_conv" + "1"
    )(inception_4a)
    inception_4e_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4e_3x3_bn" + "1"
    )(inception_4e_3x3)
    inception_4e_3x3 = layers.Activation("relu")(inception_4e_3x3)
    inception_4e_3x3 = layers.ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
    inception_4e_3x3 = layers.Conv2D(
        256, (3, 3), strides=(2, 2), name="inception_4e_3x3_conv" + "2"
    )(inception_4e_3x3)
    inception_4e_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4e_3x3_bn" + "2"
    )(inception_4e_3x3)
    inception_4e_3x3 = layers.Activation("relu")(inception_4e_3x3)

    inception_4e_5x5 = layers.Conv2D(
        64, (1, 1), strides=(1, 1), name="inception_4e_5x5_conv" + "1"
    )(inception_4a)
    inception_4e_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4e_5x5_bn" + "1"
    )(inception_4e_5x5)
    inception_4e_5x5 = layers.Activation("relu")(inception_4e_5x5)
    inception_4e_5x5 = layers.ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
    inception_4e_5x5 = layers.Conv2D(
        128, (5, 5), strides=(2, 2), name="inception_4e_5x5_conv" + "2"
    )(inception_4e_5x5)
    inception_4e_5x5 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_4e_5x5_bn" + "2"
    )(inception_4e_5x5)
    inception_4e_5x5 = layers.Activation("relu")(inception_4e_5x5)

    inception_4e_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4e_pool = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(
        inception_4e_pool
    )

    inception_4e = ops.concatenate(
        [inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=-1
    )

    # inception5a
    inception_5a_3x3 = layers.Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_5a_3x3_conv" + "1"
    )(inception_4e)
    inception_5a_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_5a_3x3_bn" + "1"
    )(inception_5a_3x3)
    inception_5a_3x3 = layers.Activation("relu")(inception_5a_3x3)
    inception_5a_3x3 = layers.ZeroPadding2D(padding=(1, 1))(inception_5a_3x3)
    inception_5a_3x3 = layers.Conv2D(
        384, (3, 3), strides=(1, 1), name="inception_5a_3x3_conv" + "2"
    )(inception_5a_3x3)
    inception_5a_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_5a_3x3_bn" + "2"
    )(inception_5a_3x3)
    inception_5a_3x3 = layers.Activation("relu")(inception_5a_3x3)

    inception_5a_pool = layers.Lambda(lambda x: x**2, name="power2_5a")(inception_4e)
    inception_5a_pool = layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(
        inception_5a_pool
    )
    inception_5a_pool = layers.Lambda(lambda x: x * 9, name="mult9_5a")(
        inception_5a_pool
    )
    inception_5a_pool = layers.Lambda(lambda x: ops.sqrt(x), name="sqrt_5a")(
        inception_5a_pool
    )

    inception_5a_pool = layers.Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_5a_pool_conv" + ""
    )(inception_5a_pool)
    inception_5a_pool = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_5a_pool_bn" + ""
    )(inception_5a_pool)
    inception_5a_pool = layers.Activation("relu")(inception_5a_pool)
    inception_5a_pool = layers.ZeroPadding2D(padding=(1, 1))(inception_5a_pool)

    inception_5a_1x1 = layers.Conv2D(
        256, (1, 1), strides=(1, 1), name="inception_5a_1x1_conv" + ""
    )(inception_4e)
    inception_5a_1x1 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_5a_1x1_bn" + ""
    )(inception_5a_1x1)
    inception_5a_1x1 = layers.Activation("relu")(inception_5a_1x1)

    inception_5a = ops.concatenate(
        [inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=-1
    )

    # inception_5b
    inception_5b_3x3 = layers.Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_5b_3x3_conv" + "1"
    )(inception_5a)
    inception_5b_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_5b_3x3_bn" + "1"
    )(inception_5b_3x3)
    inception_5b_3x3 = layers.Activation("relu")(inception_5b_3x3)
    inception_5b_3x3 = layers.ZeroPadding2D(padding=(1, 1))(inception_5b_3x3)
    inception_5b_3x3 = layers.Conv2D(
        384, (3, 3), strides=(1, 1), name="inception_5b_3x3_conv" + "2"
    )(inception_5b_3x3)
    inception_5b_3x3 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_5b_3x3_bn" + "2"
    )(inception_5b_3x3)
    inception_5b_3x3 = layers.Activation("relu")(inception_5b_3x3)

    inception_5b_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inception_5a)

    inception_5b_pool = layers.Conv2D(
        96, (1, 1), strides=(1, 1), name="inception_5b_pool_conv" + ""
    )(inception_5b_pool)
    inception_5b_pool = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_5b_pool_bn" + ""
    )(inception_5b_pool)
    inception_5b_pool = layers.Activation("relu")(inception_5b_pool)

    inception_5b_pool = layers.ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

    inception_5b_1x1 = layers.Conv2D(
        256, (1, 1), strides=(1, 1), name="inception_5b_1x1_conv" + ""
    )(inception_5a)
    inception_5b_1x1 = layers.BatchNormalization(
        axis=-1, epsilon=0.00001, name="inception_5b_1x1_bn" + ""
    )(inception_5b_1x1)
    inception_5b_1x1 = layers.Activation("relu")(inception_5b_1x1)

    inception_5b = ops.concatenate(
        [inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=-1
    )

    av_pool = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
    reshape_layer = layers.Flatten()(av_pool)
    dense_layer = layers.Dense(classes, name="dense_layer")(reshape_layer)
    norm_layer = layers.UnitNormalization(axis=1, name="norm_layer")(dense_layer)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs=inputs, outputs=norm_layer, name=name)

    if weights == "deepface":
        weights_path = utils.get_file(
            "openface.h5",
            WEIGHTS_PATH,
            cache_subdir="models",
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
