from keras import layers, Model
from keras_unet_collection.layer_utils import (
    CONV_output,
    encode_layer,
    decode_layer,
    CONV_stack,
)


def reface_encoding_block(
    X, X_skip, channel, res_num=2, activation="ReLU", batch_norm=False, name="res_conv"
):
    """
    Stacked convolutional layers with residual path.

    Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv')

    Input
    ----------
        X: input tensor.
        X_skip: the tensor that does go into the residual path
                can be a copy of X (e.g., the identity block of ResNet).
        channel: number of convolution filters.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    """
    bias_flag = not batch_norm

    for i in range(res_num):
        X = layers.Conv2D(
            channel,
            kernel_size=3,
            padding="same",
            use_bias=bias_flag,
            dilation_rate=1,
            activation=None,
            name="{}_{}".format(name, i),
        )(X)

    X = layers.add([X_skip, X], name="{}_add".format(name))

    if batch_norm:
        X = layers.BatchNormalization(axis=3, name="{}_bn".format(name))(X)

    activation_func = eval(activation)
    X = activation_func(name="{}_add_activation".format(name))(X)

    return X


def reface_decoding_block(
    X, X_skip, channel, res_num=2, activation="ReLU", batch_norm=True, name="res_decode"
):
    """
    Stacked convolutional layers with residual path.

    Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv')

    Input
    ----------
        X: input tensor.
        X_skip: the tensor that does go into the residual path
                can be a copy of X (e.g., the identity block of ResNet).
        channel: number of convolution filters.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    """

    for i in range(res_num):
        X = decode_layer(
            X,
            channel,
            kernel_size=3,
            pool_size=1,
            unpool=False,
            activation=None,
            batch_norm=None,
            name="{}_decode{}".format(name, i),
        )
        # print("layer_decode", i, X)

    # print("layer_decode_skip", X_skip)

    X = layers.add([X_skip, X], name="{}_add".format(name))

    if batch_norm:
        X = layers.BatchNormalization(axis=3, name="{}_bn".format(name))(X)

    if activation is not None:
        activation_func = eval(activation)
        X = activation_func(name="{}_add_activation".format(name))(X)

    return X


def reface_right(
    X,
    X_list,
    channel,
    d,
    activation="ReLU",
    unpool=False,
    batch_norm=True,
    name="right",
):
    """
    The decoder block of 2-d V-net.

    vnet_right(X, X_list, channel, res_num, activation='ReLU', unpool=True, batch_norm=False, name='right')

    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        d: number of residual convolutional blocks in this step
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.

    Output
    ----------
        X: output tensor.

    """
    pool_size = 2

    # print("right_in", X)

    X = decode_layer(
        X,
        channel,
        pool_size=pool_size,
        unpool=unpool,
        activation=activation,
        batch_norm=True,
        name="{}_upsample".format(name),
    )

    X = reface_decoding_block(
        X,
        X,
        channel,
        res_num=2,
        activation=activation,
        batch_norm=batch_norm,
        name="{}_decode".format(name),
    )

    # X = decode_layer(X, channel, pool_size, unpool, activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))

    # print("right0", X)

    for i in range(0, d):
        X = reface_decoding_block(
            X,
            X,
            channel,
            2,
            activation=activation,
            batch_norm=batch_norm,
            name="{}_res_conv_up_{}".format(name, i),
        )
        # print("right", i, X)

    X = layers.concatenate(
        [
            X,
        ]
        + X_list,
        axis=-1,
        name="{}_concat".format(name),
    )

    return X


def reface_left(
    X, channel, e, activation="ReLU", pool=True, batch_norm=False, name="left"
):
    """
    The encoder block of 2-d V-net.

    vnet_left(X, channel, res_num, activation='ReLU', pool=True, batch_norm=False, name='left')

    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.

    Output
    ----------
        X: output tensor.

    """

    pool_size = 2

    X = encode_layer(
        X,
        channel,
        pool_size,
        pool,
        activation=None,
        batch_norm=False,
        name="{}_encode".format(name),
    )

    for i in range(0, e):
        X = reface_encoding_block(
            X,
            X,
            channel,
            res_num=2,
            activation=activation,
            batch_norm=False,
            name="{}_res_conv_down_{}".format(name, i),
        )
    return X


def _reface_unet_base(
    input_tensor,
    filter_num: list,
    E: list,
    D: list,
    activation="ReLU",
    batch_norm=True,
    pool=False,
    unpool=False,
    name="reface",
):
    if not (len(filter_num) == len(E) == len(D)):
        raise ValueError(
            "The length of the filter list, E, and D must be equal got {}, {}, {}".format(
                len(filter_num), len(E), len(D)
            )
        )

    res_num_list = E + D

    X_skip = []

    X = input_tensor

    # print("Input", X)

    X = CONV_stack(
        X,
        filter_num[0],
        kernel_size=3,
        stack_num=1,
        dilation_rate=1,
        activation=activation,
        batch_norm=batch_norm,
        name="{}_input_conv".format(name),
    )

    X = reface_encoding_block(
        X,
        X,
        filter_num[0],
        res_num=res_num_list[0],
        activation=activation,
        batch_norm=batch_norm,
        name="{}_down_0".format(name),
    )

    X_skip.append(X)

    # print("down0", X)

    for i, f in enumerate(filter_num[1:]):
        X = reface_left(
            X,
            f,
            e=res_num_list[i + 1],
            activation=activation,
            pool=pool,
            batch_norm=batch_norm,
            name="{}_down_{}".format(name, i + 1),
        )
        # print("down", i+1, X)

        X_skip.append(X)

    X_skip = X_skip[:-1][::-1]
    filter_num = filter_num[:-1][::-1]
    res_num_list = res_num_list[:-1][::-1]

    # print("\nSkip connects", X_skip)

    # print("\nup0", X)

    # Upsampling Levels
    for i, f in enumerate(filter_num):
        X = reface_right(
            X,
            [
                X_skip[i],
            ],
            f,
            d=res_num_list[i],
            activation=activation,
            unpool=unpool,
            batch_norm=batch_norm,
            name="{}_up_{}".format(name, i),
        )
        # print("up", i + 1, X)

    return X


def reface_unet(
    input_size,
    filter_num,
    E,
    D,
    n_labels,
    activation="ReLU",
    output_activation="Softmax",
    batch_norm=True,
    pool=False,
    unpool=False,
    name="reface",
):
    """
    ReFace

    reface_unet(input_size, filter_num, E, D, n_labels,
                 aspp_num_down=256, aspp_num_up=128, activation='ReLU', output_activation='Softmax',
                 batch_norm=True, pool=True, unpool=True, name='resunet')

    ----------

    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        E: a list that defines the number of residual blocks for the encoder
        D: a list that defiens the number of residual blocks for the decoder
        n_labels: number of output labels.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        name: prefix of the created keras layers.

    Output
    ----------
        model: a keras model.
    """

    # activation_func = eval(activation)

    IN = layers.Input(input_size)
    X = IN

    X = _reface_unet_base(
        X,
        filter_num,
        E=E,
        D=D,
        batch_norm=batch_norm,
        activation=activation,
        pool=pool,
        unpool=unpool,
        name=name,
    )

    # output layer
    OUT = CONV_output(
        X,
        n_labels,
        kernel_size=1,
        activation=output_activation,
        name="{}_output".format(name),
    )

    model = Model(
        inputs=[
            IN,
        ],
        outputs=[
            OUT,
        ],
        name="{}_model".format(name),
    )

    return model
