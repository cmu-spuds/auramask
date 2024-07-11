from types import FunctionType, NoneType
from keras import layers, Model, backend, utils, KerasTensor
from auramask.layers.ResBlock import ResBlock2D


def build_res_dce_net(
    input_shape: tuple | NoneType = None,
    input_tensor: KerasTensor | NoneType = None,
    filters: int = 32,
    layer_activations: str | FunctionType = "relu",
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    block_count: int | list[int] = 2,
    block_depth=2,
    name="res-zero-dce",
):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if isinstance(block_count, int):
        block_count = [block_count] * 7

    conv1 = ResBlock2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation=layer_activations,
        basic_block_count=block_count[0],
        basic_block_depth=block_depth,
    )(img_input)

    # print(conv1)

    conv2 = ResBlock2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation=layer_activations,
        basic_block_count=block_count[1],
        basic_block_depth=block_depth,
    )(conv1)

    # print(conv2)

    conv3 = ResBlock2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation=layer_activations,
        basic_block_count=block_count[2],
        basic_block_depth=block_depth,
    )(conv2)

    # print(conv3)

    conv4 = ResBlock2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation=layer_activations,
        basic_block_count=block_count[3],
        basic_block_depth=block_depth,
    )(conv3)

    # print(conv4)

    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = ResBlock2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation=layer_activations,
        basic_block_count=block_count[4],
        basic_block_depth=block_depth,
    )(int_con1)

    # print(conv5)

    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = ResBlock2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=layer_activations,
        basic_block_count=block_count[5],
        basic_block_depth=block_depth,
    )(int_con2)

    # print(conv6)

    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = ResBlock2D(
        filters=24,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=None,
        basic_block_count=block_count[6],
        basic_block_depth=block_depth,
    )(int_con3)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return Model(
        inputs=inputs,
        outputs=[x_r],
        name=name,
    )
