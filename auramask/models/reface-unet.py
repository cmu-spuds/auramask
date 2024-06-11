# from keras import layers, Model
# from keras_unet_collection.layer_utils import CONV_stack, CONV_output


# def ReFace_block(
#     X, channel, kernel_size=3, activation="ReLU", batch_norm=False, name="reface_block"
# ):
#     """
#     ReFace_block

#     ----------
#     Input
#     ----------
#         X: input tensor
#         channel: number of convolution filters
#         kernel_size: size of 2-d convolution kernels
#         activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
#         batch_norm: True for batch normalization, False otherwise.
#         name: name of the created keras layers
#     Output
#     ----------
#         X: output tensor

#     """

#     x = CONV_stack(
#         X,
#         channel,
#         kernel_size=kernel_size,
#         stack_num=2,
#         dilation_rate=1,
#         activation=activation,
#         batch_norm=batch_norm,
#         name=name,
#     )

#     X = layers.add([X, x], name="{}_add".format(name))

#     return X


# def decode_layer(
#     X,
#     channel,
#     pool_size=2,
#     kernel_size=3,
#     activation="ReLU",
#     batch_norm=False,
#     name="decode",
# ):
#     """
#     An overall decode layer, based on trans conv.

#     decode_layer(X, channel, pool_size, unpool, kernel_size=3,
#                  activation='ReLU', batch_norm=False, name='decode')

#     Input
#     ----------
#         X: input tensor.
#         pool_size: the decoding factor.
#         channel: (for trans conv only) number of convolution filters.
#         kernel_size: size of convolution kernels.
#                      If kernel_size='auto', then it equals to the `pool_size`.
#         activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
#         batch_norm: True for batch normalization, False otherwise.
#         name: prefix of the created keras layers.

#     Output
#     ----------
#         X: output tensor.

#     * The defaut: `kernel_size=3`, is suitable for `pool_size=2`.

#     """
#     if kernel_size == "auto":
#         kernel_size = pool_size

#     x = layers.Conv2DTranspose(
#         channel,
#         kernel_size,
#         strides=(pool_size, pool_size),
#         padding="same",
#         name="{}_trans_conv0".format(name),
#     )(X)

#     x = layers.Conv2DTranspose(
#         channel,
#         kernel_size,
#         strides=(pool_size, pool_size),
#         padding="same",
#         name="{}_trans_conv1".format(name),
#     )(x)

#     # batch normalization
#     if batch_norm:
#         x = layers.BatchNormalization(axis=3, name="{}_bn".format(name))(x)

#     # activation
#     if activation is not None:
#         activation_func = eval(activation)
#         x = activation_func(name="{}_activation".format(name))(x)

#     X = layers.add([X, x], name="{}_add".format(name))

#     return X


# def encode_layer(
#     X,
#     channel,
#     pool_size,
#     kernel_size="auto",
#     activation="ReLU",
#     batch_norm=False,
#     name="encode",
# ):
#     """
#     An overall encode layer, based on one of the:
#     (1) max-pooling, (2) average-pooling, (3) strided conv2d.

#     encode_layer(X, channel, pool_size, pool, kernel_size='auto',
#                  activation='ReLU', batch_norm=False, name='encode')

#     Input
#     ----------
#         X: input tensor.
#         pool_size: the encoding factor.
#         channel: (for strided conv only) number of convolution filters.
#         pool: True or 'max' for MaxPooling2D.
#               'ave' for AveragePooling2D.
#               False for strided conv + batch norm + activation.
#         kernel_size: size of convolution kernels.
#                      If kernel_size='auto', then it equals to the `pool_size`.
#         activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
#         batch_norm: True for batch normalization, False otherwise.
#         name: prefix of the created keras layers.

#     Output
#     ----------
#         X: output tensor.

#     """
#     bias_flag = not batch_norm

#     if kernel_size == "auto":
#         kernel_size = pool_size

#     # linear convolution with strides
#     x = layers.Conv2D(
#         channel,
#         kernel_size,
#         strides=(pool_size, pool_size),
#         padding="valid",
#         use_bias=bias_flag,
#         name="{}_stride_conv".format(name),
#     )(X)

#     x = layers.Conv2D(
#         channel,
#         kernel_size,
#         strides=(pool_size, pool_size),
#         padding="valid",
#         use_bias=bias_flag,
#         name="{}_stride_conv".format(name),
#     )(x)

#     # batch normalization
#     if batch_norm:
#         x = layers.BatchNormalization(axis=3, name="{}_bn".format(name))(x)

#     # activation
#     if activation is not None:
#         activation_func = eval(activation)
#         x = activation_func(name="{}_activation".format(name))(x)

#     X = layers.add([X, x], name="{}_add".format(name))

#     return X


# def REFACE_right(
#     X,
#     X_list,
#     channel,
#     kernel_size=3,
#     stack_num=2,
#     activation="ReLU",
#     unpool=True,
#     batch_norm=False,
#     concat=True,
#     name="right0",
# ):
#     """
#     The decoder block of ReFace.

#     Input
#     ----------
#         X: input tensor.
#         X_list: a list of other tensors that connected to the input tensor.
#         channel: number of convolution filters.
#         kernel_size: size of 2-d convolution kernels.
#         stack_num: number of convolutional layers.
#         activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
#         unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
#                 'nearest' for Upsampling2D with nearest interpolation.
#                 False for Conv2DTranspose + batch norm + activation.
#         batch_norm: True for batch normalization, False otherwise.
#         concat: True for concatenating the corresponded X_list elements.
#         name: prefix of the created keras layers.

#     Output
#     ----------
#         X: output tensor.

#     """

#     pool_size = 2

#     X = decode_layer(
#         X,
#         channel,
#         pool_size,
#         unpool,
#         activation=activation,
#         batch_norm=batch_norm,
#         name="{}_decode".format(name),
#     )

#     if concat:
#         # <--- *stacked convolutional can be applied here
#         X = layers.concatenate(
#             [
#                 X,
#             ]
#             + X_list,
#             axis=3,
#             name=name + "_concat",
#         )

#     # Stacked convolutions after concatenation
#     X = CONV_stack(
#         X,
#         channel,
#         kernel_size,
#         stack_num=stack_num,
#         activation=activation,
#         batch_norm=batch_norm,
#         name=name + "_conv_after_concat",
#     )

#     return X


# def reface_unet(
#     input_size,
#     filter_num,
#     E,
#     D,
#     n_labels,
#     activation="ReLU",
#     output_activation="Softmax",
#     batch_norm=True,
#     pool=True,
#     unpool=True,
#     name="resunet",
# ):
#     """
#     ReFace

#     reface_unet(input_size, filter_num, E, D, n_labels,
#                  aspp_num_down=256, aspp_num_up=128, activation='ReLU', output_activation='Softmax',
#                  batch_norm=True, pool=True, unpool=True, name='resunet')

#     ----------

#     Input
#     ----------
#         input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
#         filter_num: a list that defines the number of filters for each \
#                     down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
#                     The depth is expected as `len(filter_num)`.
#         E: a list that defines the number of residual blocks for the encoder
#         D: a list that defiens the number of residual blocks for the decoder
#         n_labels: number of output labels.
#         activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
#         output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
#                            Default option is 'Softmax'.
#                            if None is received, then linear activation is applied.
#         batch_norm: True for batch normalization.
#         unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
#                 'nearest' for Upsampling2D with nearest interpolation.
#                 False for Conv2DTranspose + batch norm + activation.
#         name: prefix of the created keras layers.

#     Output
#     ----------
#         model: a keras model.
#     """

#     # activation_func = eval(activation)

#     IN = layers.Input(input_size)
#     X = IN
#     X_skip = []

#     # downsampling blocks
#     X = REFACE_stack(
#         X,
#         filter_num[0],
#         stack_num=E[0],
#         activation=activation,
#         batch_norm=batch_norm,
#         name="{}_down0".format(name),
#     )
#     X_skip.append(X)

#     for i, f in enumerate(filter_num[1:]):
#         X = REFACE_left(
#             X,
#             f,
#             stack_num=E[i],
#             activation=activation,
#             pool=pool,
#             batch_norm=batch_norm,
#             name="{}_down{}".format(name, i + 1),
#         )
#         X_skip.append(X)

#     # upsampling blocks
#     X_skip = X_skip[:-1][::-1]
#     for i, f in enumerate(filter_num[:-1][::-1]):
#         X = REFACE_right(
#             X,
#             [
#                 X_skip[i],
#             ],
#             f,
#             stack_num=D[i],
#             activation=activation,
#             unpool=unpool,
#             batch_norm=batch_norm,
#             name="{}_up{}".format(name, i + 1),
#         )

#     OUT = CONV_output(
#         X,
#         n_labels,
#         kernel_size=1,
#         activation=output_activation,
#         name="{}_output".format(name),
#     )
#     model = Model(inputs=[IN], outputs=[OUT], name="{}_model".format(name))

#     return model
