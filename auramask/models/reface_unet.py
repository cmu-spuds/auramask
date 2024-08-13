from types import NoneType
from keras import layers, Model, backend, KerasTensor, utils
from auramask.layers.ResBlock import ResBlock2D, ResBlock2DTranspose


def reface_unet(
    filter_num: int | list[int],
    E: list[int],
    D: list[int],
    n_labels: int,
    input_tensor: KerasTensor | NoneType = None,
    input_shape: tuple | NoneType = None,
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
        activation: one of the `keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        kernel_reg: Convolutional kernel regulizer.
        name: prefix of the created keras layers.

    Output
    ----------
        model: a keras model.
    """
    channel_axis = 1 if backend.image_data_format() == "channels_first" else 3

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name="{}_input".format(name))
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(
                tensor=input_tensor, shape=input_shape, name="{}_input".format(name)
            )
        else:
            img_input = input_tensor

    if not (len(filter_num) == len(E) == len(D)):
        raise ValueError(
            "The length of the filter list, E, and D must be equal got {}, {}, {}".format(
                len(filter_num), len(E), len(D)
            )
        )

    X_skip = []

    X = img_input

    for i, f in enumerate(filter_num):
        if batch_norm:
            X = layers.BatchNormalization(
                axis=channel_axis, name="{}_down_{}_bn".format(name, i)
            )(X)
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

    filter_num = filter_num[::-1]

    # Upsampling Levels
    for i, f in enumerate(filter_num):
        if batch_norm:
            X = layers.BatchNormalization(
                axis=channel_axis, name="{}_up_{}_bn".format(name, i)
            )(X)
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
                axis=channel_axis,
                name="{}_up_{}_concat".format(name, i),
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

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(
        inputs=inputs,
        outputs=[
            X,
        ],
        name="{}_model".format(name),
    )

    return model
