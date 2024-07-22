from types import FunctionType, NoneType
from keras import layers, Model, backend, utils, KerasTensor
from auramask.layers.ResBlock import ResBlock2D, ResBlock2DTranspose


def build_res_dce_net(
    input_shape: tuple | NoneType = None,
    input_tensor: KerasTensor | NoneType = None,
    filters: int | list = 32,
    layer_activations: str | FunctionType = "relu",
    kernel_size=(3, 3),
    strides=(1, 1),
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
        name="conv_{}_down".format(0),
    )(img_input)
    if batch_norm:
        x = layers.BatchNormalization(epsilon=1e-5, name="conv_{}_down_bn".format(0))(x)

    x_skip = [x]

    depth = len(filters)

    for i in range(1, depth):
        x = ResBlock2D(
            filters=filters[i],
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            activation=layer_activations,
            kernel_regularizer=kernel_regularizer,
            basic_block_count=block_count[i],
            basic_block_depth=block_depth,
            name="conv_{}_down".format(i),
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(
                epsilon=1e-5, name="conv_{}_down_bn".format(i)
            )(x)
        x_skip.append(x)

    if unet:
        x = x_skip.pop()

    filters = filters[::-1]

    for i in range(0, depth):
        if unet and i != 0:
            x = ResBlock2DTranspose(
                filters=filters[i],
                kernel_size=kernel_size,
                padding=padding,
                strides=strides,
                activation=layer_activations,
                kernel_regularizer=kernel_regularizer,
                basic_block_count=block_count[depth + i],
                basic_block_depth=block_depth,
                name="convt_{}_up".format(depth - i),
            )(x)
            if batch_norm:
                x = layers.BatchNormalization(
                    epsilon=1e-5, name="convt_{}_up_bn".format(depth - i)
                )(x)
            x = layers.Concatenate(
                axis=-1, name="convt_{}_up_concat".format(depth - i)
            )([x, x_skip.pop()])
            x = ResBlock2D(
                filters=filters[i],
                basic_block_depth=2,
                activation="relu",
                name="convt_{}_up_post_concat_conv".format(depth - i),
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
