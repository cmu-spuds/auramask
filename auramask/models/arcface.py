from keras import Model, utils, layers, backend, ops
from keras.src.applications.imagenet_utils import obtain_input_shape
import os

WEIGHTS_PATH = "https://huggingface.co/logasja/ArcFace/resolve/main/" "model.weights.h5"


def preprocess_input(x):
    x = ops.subtract(x, 127.5)
    x = ops.divide(x, 128)
    return x


def ResNet(
    stack_fn,
    model_name="resnet",
    input_tensor=None,
    input_shape=None,
    image_data_format=None,
    **kwargs,
):
    """Instantiates the ResNet

    Args:
      stack_fn: a function that returns output tensor for the
        stacked residual blocks.
      model_name: string, model name.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
    Returns:
      A `keras.Model` instance.
    """
    if not image_data_format:
        image_data_format = backend.image_data_format()

    # Determine proper input shape
    input_shape = obtain_input_shape(
        input_shape,
        default_size=112,
        min_size=32,
        data_format=image_data_format,
        require_flatten=True,
        weights="imagenet",
    )

    shared_axes = [1, 2] if image_data_format == "channels_last" else [2, 3]

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if image_data_format == "channels_last" else 1

    x = layers.ZeroPadding2D(padding=1, name="conv1_pad")(img_input)
    x = layers.Conv2D(
        64,
        3,
        strides=1,
        use_bias=False,
        kernel_initializer="glorot_normal",
        name="conv1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=2e-5, momentum=0.9, name="conv1_bn"
    )(x)
    x = layers.PReLU(shared_axes=shared_axes, name="conv1_prelu")(x)
    x = stack_fn(x)

    return x


def ResNet34(
    input_tensor=None,
    input_shape=None,
):
    """Instantiates the ResNet32 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 6, name="conv4")
        return stack1(x, 512, 3, name="conv5")

    return ResNet(
        stack_fn=stack_fn,
        model_name="resnet34",
        input_tensor=input_tensor,
        input_shape=input_shape,
    )


def ArcFace(
    include_top=True,
    weights="deepface",
    input_tensor=None,
    input_shape=None,
    classes=512,
    classifier_activation="softmax",
    preprocess=False,
    name="ArcFace",
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

    if weights == "deepface" and include_top and classes != 512:
        raise ValueError(
            'If using `weights` as `"deepface"` with `include_top` '
            "as true, `classes` should be 128 or 512.  "
            f"Received: `classes={classes}.`"
        )

    if not image_data_format:
        image_data_format = backend.image_data_format()

    # Determine proper input shape
    input_shape = obtain_input_shape(
        input_shape,
        default_size=112,
        min_size=75,
        data_format=image_data_format,
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

    x = ResNet34(input_tensor=x)

    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(
        classes, activation=None, use_bias=True, kernel_initializer="glorot_normal"
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9, epsilon=2e-5, name="embedding", scale=True
    )(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if weights == "deepface":
        weights_path = utils.get_file(
            "arcface.weights.h5",
            WEIGHTS_PATH,
            cache_subdir="models",
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def block1(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=True,
    name=None,
    image_data_format=backend.image_data_format(),
):
    bn_axis = 3 if image_data_format == "channels_last" else 1
    shared_axes = [1, 2] if image_data_format == "channels_last" else [2, 3]

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            kernel_initializer="glorot_normal",
            name=name + "_0_conv",
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_1_bn"
    )(x)
    x = layers.ZeroPadding2D(padding=1, name=name + "_1_pad")(x)
    x = layers.Conv2D(
        filters,
        3,
        strides=1,
        kernel_initializer="glorot_normal",
        use_bias=False,
        name=name + "_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_2_bn"
    )(x)
    x = layers.PReLU(shared_axes=shared_axes, name=name + "_1_prelu")(x)

    x = layers.ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        kernel_initializer="glorot_normal",
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_3_bn"
    )(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    x = block1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
    return x
