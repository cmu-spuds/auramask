from types import FunctionType, NoneType
from keras import layers, Model, backend, utils, ops, KerasTensor


def build_dce_net(
    input_shape: tuple | NoneType = None,
    input_tensor: KerasTensor | NoneType = None,
    filters: int = 32,
    layer_activations: str | FunctionType = "relu",
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    name="zero-dce",
):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    conv1 = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        activation=layer_activations,
        padding=padding,
    )(img_input)
    conv2 = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        activation=layer_activations,
        padding=padding,
    )(conv1)
    conv3 = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        activation=layer_activations,
        padding=padding,
    )(conv2)
    conv4 = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        activation=layer_activations,
        padding=padding,
    )(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        activation=layer_activations,
        padding=padding,
    )(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        activation=layer_activations,
        padding=padding,
    )(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(
        24, kernel_size, strides=strides, activation=None, padding=padding
    )(int_con3)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return Model(
        inputs=inputs,
        outputs=[
            x_r,
        ],
        name=name,
    )


def get_enhanced_image(output, data):
    r = ops.split(output, 8, -1)
    x = data + r[0] * (ops.square(data) - data)
    x = x + r[1] * (ops.square(x) - x)
    x = x + r[2] * (ops.square(x) - x)
    enhanced_image = x + r[3] * (ops.square(x) - x)
    x = enhanced_image + r[4] * (ops.square(enhanced_image) - enhanced_image)
    x = x + r[5] * (ops.square(x) - x)
    x = x + r[6] * (ops.square(x) - x)
    enhanced_image = x + r[7] * (ops.square(x) - x)
    return enhanced_image, output
