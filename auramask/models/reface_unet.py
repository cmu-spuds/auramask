from keras import layers, Model
from auramask.layers.ResBlock import ResBlock2D, ResBlock2DTranspose


def _reface_unet_base(
    input_tensor,
    filter_num: list,
    E: list,
    D: list,
    activation="relu",
    batch_norm=True,
    kernel_reg="l2",
    name="reface",
):
    if not (len(filter_num) == len(E) == len(D)):
        raise ValueError(
            "The length of the filter list, E, and D must be equal got {}, {}, {}".format(
                len(filter_num), len(E), len(D)
            )
        )

    X_skip = []

    X = input_tensor

    for i, f in enumerate(filter_num):
        if batch_norm:
            X = layers.BatchNormalization(axis=3, name="{}_down_{}_bn".format(name, i))(
                X
            )
        X = ResBlock2D(
            f,
            basic_block_depth=2,
            basic_block_count=E[i],
            kernel_regularizer=kernel_reg,
            activation=activation,
            name="{}_down_{}".format(name, i),
        )(X)
        X_skip.append(X)

    X = X_skip.pop()

    X_skip = X_skip[::-1]
    filter_num = filter_num[::-1]

    # Upsampling Levels
    for i, f in enumerate(filter_num):
        if batch_norm:
            X = layers.BatchNormalization(axis=3, name="{}_up_{}_bn".format(name, i))(X)
        X = ResBlock2DTranspose(
            filter_num[i],
            basic_block_depth=2,
            basic_block_count=D[i],
            kernel_regularizer=kernel_reg,
            name="{}_up_{}".format(name, i),
        )(X)
        if len(X_skip) > 0:
            skip_conn = X_skip.pop()
            X = layers.concatenate(
                [X, skip_conn],
                axis=-1,
                name="{}_up_{}_concat".format(name, i),
            )

    return X


def reface_unet(
    input_shape: tuple,
    filter_num: int | list[int],
    E: list[int],
    D: list[int],
    n_labels: int,
    activation="relu",
    output_activation="softmax",
    batch_norm=True,
    kernel_reg=None,
    name="reface",
):
    """
    ReFace

    reface_unet(input_shape, filter_num, E, D, n_labels,
                 activation='relu', output_activation='softmax',
                 batch_norm=True, kernel_reg="l2", name='resunet')

    ----------

    Input
    ----------
        input_shape: the size/shape of network input, e.g., `(128, 128, 3)`.
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
        kernel_reg: Convolutional kernel regulizer.
        name: prefix of the created keras layers.

    Output
    ----------
        model: a keras model.
    """
    IN = layers.Input(input_shape)

    X = _reface_unet_base(
        IN,
        filter_num,
        E=E,
        D=D,
        batch_norm=batch_norm,
        activation=activation,
        kernel_reg=kernel_reg,
        name=name,
    )

    X = layers.Conv2D(
        n_labels,
        1,
        padding="same",
        use_bias=True,
        kernel_regularizer=kernel_reg,
        name="{}_out_conv".format(name),
    )(X)

    X = layers.Activation(output_activation)(X)

    model = Model(
        inputs=[
            IN,
        ],
        outputs=[
            X,
        ],
        name="{}_model".format(name),
    )

    return model
