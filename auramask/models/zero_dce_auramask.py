from types import FunctionType, NoneType
from typing import Literal
from keras import layers, Model, backend, utils, KerasTensor
from auramask.layers.ResBlock import ResBlock2D, ResBlock2DTranspose


def build_res_dce_net(
    input_shape: tuple | NoneType = None,
    input_tensor: KerasTensor | NoneType = None,
    filters: int | list = 32,
    layer_activations: str | FunctionType = "relu",
    pooling: bool | Literal["max"] | Literal["avg"] = False,
    unpooling: bool
    | Literal["nearest"]
    | Literal["bilinear"]
    | Literal["bicubic"]
    | Literal["lanczos3"]
    | Literal["lanczos5"] = False,
    kernel_size=(3, 3),
    padding="same",
    block_count: int | list[int] = 2,
    block_depth=2,
    kernel_regularizer=None,
    batch_norm=False,
    unet=False,
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
        block_count = [block_count] * 6

    if unet and isinstance(filters, int):
        filters = [filters * (2**i) for i in range(3)]
    elif not unet:
        filters = [filters] * 3
    else:
        raise Exception("Invalid Arguments")

    if batch_norm:
        img_input = layers.BatchNormalization(epsilon=1e-5)(img_input)

    x = ResBlock2D(
        filters=filters[0],
        kernel_size=kernel_size,
        padding=padding,
        strides=(1, 1),
        activation=layer_activations,
        kernel_regularizer=kernel_regularizer,
        basic_block_count=block_count[0],
        basic_block_depth=block_depth,
        name="down_{}_conv".format(0),
    )(img_input)
    if batch_norm:
        x = layers.BatchNormalization(epsilon=1e-5, name="down_{}_bn".format(0))(x)

    x_skip = [x]

    depth = len(filters)

    for i in range(1, depth):
        strides = (1, 1)
        if not pooling:
            strides = (2, 2)
        elif pooling == "max" or pooling is True:
            x = layers.MaxPool2D(name="down_{}_maxpool".format(i))(x)
        elif pooling == "avg":
            x = layers.AveragePooling2D(2, name="down_{}_avgpool".format(i))(x)
        else:
            raise Exception("Invalid pooling argument: {}".format(pooling))

        x = ResBlock2D(
            filters=filters[i],
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            activation=layer_activations,
            kernel_regularizer=kernel_regularizer,
            basic_block_count=block_count[i],
            basic_block_depth=block_depth,
            name="down_{}_conv".format(i),
        )(x)

        if batch_norm:
            x = layers.BatchNormalization(epsilon=1e-5, name="down_{}_bn".format(i))(x)
        x_skip.append(x)

    if unet:
        x = x_skip.pop()

    filters = filters[::-1]

    for i in range(0, depth):
        if unet and i != 0:
            stride = (1, 1)
            if not unpooling:
                stride = (2, 2)
                x = ResBlock2DTranspose(
                    filters=filters[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    strides=stride,
                    activation=layer_activations,
                    kernel_regularizer=kernel_regularizer,
                    basic_block_count=block_count[depth + i],
                    basic_block_depth=block_depth,
                    name="up_{}_convt".format(depth - i),
                )(x)
            elif unpooling or unpooling == "bilinear":
                interp = unpooling if isinstance(unpooling, str) else "bilinear"
                x = layers.UpSampling2D(
                    size=(2, 2),
                    interpolation=interp,
                    name="up_{}_unpool".format(depth - i),
                )(x)
            else:
                raise Exception("Invalid unpooling argument: {}".format(unpooling))

            if batch_norm:
                x = layers.BatchNormalization(
                    epsilon=1e-5, name="up_{}_bn".format(depth - i)
                )(x)
            x = layers.Concatenate(axis=-1, name="up_{}_concat".format(depth - i))(
                [x, x_skip.pop()]
            )
            x = ResBlock2D(
                filters=filters[i],
                kernel_size=kernel_size,
                basic_block_depth=2 if unpooling is False else block_depth,
                basic_block_count=1 if unpooling is False else block_count[depth + i],
                kernel_regularizer=kernel_regularizer,
                activation=layer_activations,
                name="up_{}_conv".format(depth - i),
            )(x)
        elif unet and i == 0:
            continue
        else:
            x = ResBlock2D(
                filters=filters[i],
                kernel_size=kernel_size,
                padding=padding,
                strides=strides,
                activation=layer_activations,
                kernel_regularizer=kernel_regularizer,
                basic_block_count=block_count[depth + i],
                basic_block_depth=block_depth,
                name="convt_{}_up".format(i),
            )(x)
            if batch_norm:
                x = layers.BatchNormalization(
                    epsilon=1e-5, name="convt_{}_up_bn".format(i)
                )(x)
            x = layers.Concatenate(axis=-1)([x, x_skip.pop()])

    x_r = ResBlock2D(
        24,
        activation=None,
        padding=padding,
        basic_block_depth=1,
        basic_block_count=1,
        kernel_regularizer=kernel_regularizer,
    )(x)

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
