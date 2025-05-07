from keras import Model, utils, layers, backend, ops, KerasTensor
from keras.src.applications.imagenet_utils import obtain_input_shape
import os
from functools import partial

WEIGHTS_PATH_128 = (
    "https://huggingface.co/logasja/FaceNet/resolve/main/" "model.weights.h5"
)

WEIGHTS_PATH_512 = (
    "https://huggingface.co/logasja/FaceNet512/resolve/main/" "model.weights.h5"
)


def preprocess_input(x) -> KerasTensor:
    mean, std = (
        ops.mean(x, axis=[-3, -2, -1], keepdims=True),
        ops.std(x, axis=[-3, -2, -1], keepdims=True),
    )
    x = ops.subtract(x, mean)
    x = ops.divide(x, ops.maximum(std, backend.epsilon()))
    return x


def FaceNet(
    include_top=True,
    weights="deepface",
    input_tensor=None,
    input_shape=None,
    classes=128,
    classifier_activation="softmax",
    preprocess=False,
    name="FaceNet",
) -> Model:
    if not (weights in {"deepface", None} or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `deepface` "
            "(pre-training on VGG-Face2 (serengil deepface repo)), "
            "or the path to the weights file to be loaded.  "
            f"Received: `weights={weights}.`"
        )

    if weights == "deepface" and include_top and classes != 128 and classes != 512:
        raise ValueError(
            'If using `weights` as `"deepface"` with `include_top` '
            "as true, `classes` should be 128 or 512.  "
            f"Received: `classes={classes}.`"
        )

    # Determine proper input shape
    input_shape = obtain_input_shape(
        input_shape,
        default_size=160,
        min_size=75,
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

    x = InceptionResNetV1(x, classes=classes)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if weights == "deepface" and classes == 128:
        weights_path = utils.get_file(
            "facenet128.weights.h5",
            WEIGHTS_PATH_128,
            cache_subdir="models",
        )
        model.load_weights(weights_path)
    elif weights == "deepface" and classes == 512:
        weights_path = utils.get_file(
            "facenet512.weights.h5",
            WEIGHTS_PATH_512,
            cache_subdir="models",
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


"""Inception-ResNet V1 model for Keras.
From https://github.com/nyoki-mtl/keras-facenet/blob/master/code/inception_resnet_v1.py
# Reference
http://arxiv.org/abs/1602.07261
https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py
https://github.com/myutwo150/keras-inception-resnet-v2/blob/master/inception_resnet_v2.py
"""


def scaling(x, scale):
    return x * scale


def _generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return "_".join((prefix, name))
    return "_".join((prefix, "Branch", str(branch_idx), name))


def conv2d_bn(
    x,
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
    use_bias=False,
    name=None,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name,
    )(x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == "channels_first" else 3
        bn_name = _generate_layer_name("BatchNorm", prefix=name)
        x = layers.BatchNormalization(
            axis=bn_axis, momentum=0.995, epsilon=0.001, scale=False, name=bn_name
        )(x)
    if activation is not None:
        ac_name = _generate_layer_name("Activation", prefix=name)
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation="relu"):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
    if block_idx is None:
        prefix = None
    else:
        prefix = "_".join((block_type, str(block_idx)))
    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == "Block35":
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt("Conv2d_1x1", 0))
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt("Conv2d_0a_1x1", 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt("Conv2d_0b_3x3", 1))
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt("Conv2d_0a_1x1", 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt("Conv2d_0b_3x3", 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt("Conv2d_0c_3x3", 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == "Block17":
        branch_0 = conv2d_bn(x, 128, 1, name=name_fmt("Conv2d_1x1", 0))
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt("Conv2d_0a_1x1", 1))
        branch_1 = conv2d_bn(branch_1, 128, [1, 7], name=name_fmt("Conv2d_0b_1x7", 1))
        branch_1 = conv2d_bn(branch_1, 128, [7, 1], name=name_fmt("Conv2d_0c_7x1", 1))
        branches = [branch_0, branch_1]
    elif block_type == "Block8":
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt("Conv2d_1x1", 0))
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt("Conv2d_0a_1x1", 1))
        branch_1 = conv2d_bn(branch_1, 192, [1, 3], name=name_fmt("Conv2d_0b_1x3", 1))
        branch_1 = conv2d_bn(branch_1, 192, [3, 1], name=name_fmt("Conv2d_0c_3x1", 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError(
            "Unknown Inception-ResNet block type. "
            'Expects "Block35", "Block17" or "Block8", '
            "but got: " + str(block_type)
        )

    mixed = layers.Concatenate(axis=channel_axis, name=name_fmt("Concatenate"))(
        branches
    )
    up = conv2d_bn(
        mixed,
        ops.shape(x)[channel_axis],
        1,
        activation=None,
        use_bias=True,
        name=name_fmt("Conv2d_1x1"),
    )
    up = layers.Lambda(
        scaling, output_shape=ops.shape(up)[1:], arguments={"scale": scale}
    )(up)
    x = layers.add([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=name_fmt("Activation"))(x)
    return x


def InceptionResNetV1(input_tensor, classes=128, dropout_keep_prob=0.8):
    x = conv2d_bn(input_tensor, 32, 3, strides=2, padding="valid", name="Conv2d_1a_3x3")
    x = conv2d_bn(x, 32, 3, padding="valid", name="Conv2d_2a_3x3")
    x = conv2d_bn(x, 64, 3, name="Conv2d_2b_3x3")
    x = layers.MaxPooling2D(3, strides=2, name="MaxPool_3a_3x3")(x)
    x = conv2d_bn(x, 80, 1, padding="valid", name="Conv2d_3b_1x1")
    x = conv2d_bn(x, 192, 3, padding="valid", name="Conv2d_4a_3x3")
    x = conv2d_bn(x, 256, 3, strides=2, padding="valid", name="Conv2d_4b_3x3")

    # 5x Block35 (Inception-ResNet-A block):
    for block_idx in range(1, 6):
        x = inception_resnet_block(
            x, scale=0.17, block_type="Block35", block_idx=block_idx
        )

    # Mixed 6a (Reduction-A block):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
    name_fmt = partial(_generate_layer_name, prefix="Mixed_6a")
    branch_0 = conv2d_bn(
        x, 384, 3, strides=2, padding="valid", name=name_fmt("Conv2d_1a_3x3", 0)
    )
    branch_1 = conv2d_bn(x, 192, 1, name=name_fmt("Conv2d_0a_1x1", 1))
    branch_1 = conv2d_bn(branch_1, 192, 3, name=name_fmt("Conv2d_0b_3x3", 1))
    branch_1 = conv2d_bn(
        branch_1, 256, 3, strides=2, padding="valid", name=name_fmt("Conv2d_1a_3x3", 1)
    )
    branch_pool = layers.MaxPooling2D(
        3, strides=2, padding="valid", name=name_fmt("MaxPool_1a_3x3", 2)
    )(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name="Mixed_6a")(branches)

    # 10x Block17 (Inception-ResNet-B block):
    for block_idx in range(1, 11):
        x = inception_resnet_block(
            x, scale=0.1, block_type="Block17", block_idx=block_idx
        )

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    name_fmt = partial(_generate_layer_name, prefix="Mixed_7a")
    branch_0 = conv2d_bn(x, 256, 1, name=name_fmt("Conv2d_0a_1x1", 0))
    branch_0 = conv2d_bn(
        branch_0, 384, 3, strides=2, padding="valid", name=name_fmt("Conv2d_1a_3x3", 0)
    )
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt("Conv2d_0a_1x1", 1))
    branch_1 = conv2d_bn(
        branch_1, 256, 3, strides=2, padding="valid", name=name_fmt("Conv2d_1a_3x3", 1)
    )
    branch_2 = conv2d_bn(x, 256, 1, name=name_fmt("Conv2d_0a_1x1", 2))
    branch_2 = conv2d_bn(branch_2, 256, 3, name=name_fmt("Conv2d_0b_3x3", 2))
    branch_2 = conv2d_bn(
        branch_2, 256, 3, strides=2, padding="valid", name=name_fmt("Conv2d_1a_3x3", 2)
    )
    branch_pool = layers.MaxPooling2D(
        3, strides=2, padding="valid", name=name_fmt("MaxPool_1a_3x3", 3)
    )(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name="Mixed_7a")(branches)

    # 5x Block8 (Inception-ResNet-C block):
    for block_idx in range(1, 6):
        x = inception_resnet_block(
            x, scale=0.2, block_type="Block8", block_idx=block_idx
        )
    x = inception_resnet_block(
        x, scale=1.0, activation=None, block_type="Block8", block_idx=6
    )

    # Classification block
    x = layers.GlobalAveragePooling2D(name="AvgPool")(x)
    x = layers.Dropout(1.0 - dropout_keep_prob, name="Dropout")(x)
    # Bottleneck
    x = layers.Dense(classes, use_bias=False, name="Bottleneck")(x)
    bn_name = _generate_layer_name("BatchNorm", prefix="Bottleneck")
    x = layers.BatchNormalization(
        momentum=0.995, epsilon=0.001, scale=False, name=bn_name
    )(x)

    return x
