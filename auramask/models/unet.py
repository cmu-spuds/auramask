from keras import layers, Model


class EncoderBlock(layers.Layer):
    def __init__(
        self,
        n_filters=32,
        kernel=3,
        dropout_prob=0.3,
        max_pooling=True,
        name="UEncoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.dropout_prob = dropout_prob
        self.max_pooling = max_pooling

        self.conv1 = layers.Conv2D(
            n_filters,
            kernel,
            activation="relu",
            padding="same",
            kernel_initializer="HeNormal",
        )
        self.conv2 = layers.Conv2D(
            n_filters,
            kernel,
            activation="relu",
            padding="same",
            kernel_initializer="HeNormal",
        )

        self.bn = layers.BatchNormalization()
        self.do = layers.Dropout(dropout_prob) if dropout_prob > 0 else None
        self.mp = layers.MaxPooling2D(pool_size=(2, 2)) if max_pooling else None

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Batch norm to normlaize output of last layer based on batch mean and std
        x = self.bn(x, training=False)

        # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
        if self.do:
            x = self.do(x)

        # Pooling reduces the size of the image while keeping the number of channels same
        # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
        # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
        if self.mp:
            next_layer = self.mp(x)
        else:
            next_layer = x

        return next_layer, x


class DecoderBlock(layers.Layer):
    def __init__(self, kernel=3, stride=2, n_filters=32, name="UDecoder", **kwargs):
        super().__init__(name=name, **kwargs)

        self.up1 = layers.Conv2DTranspose(
            n_filters,
            (kernel, kernel),  # Kernel size
            strides=(stride, stride),
            padding="same",
        )

        self.conv1 = layers.Conv2D(
            n_filters,
            kernel,
            activation="relu",
            padding="same",
            kernel_initializer="HeNormal",
        )

        self.conv2 = layers.Conv2D(
            n_filters,
            kernel,
            activation="relu",
            padding="same",
            kernel_initializer="HeNormal",
        )

    def call(self, x):
        x, skp = x

        # Start with a transpose convolution layer to first increase the size of the image
        x = self.up1(x)

        # Merge the skip connection from previous block to prevent information loss
        x = layers.concatenate([x, skp], axis=3)

        # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
        # The parameters for the function are similar to encoder
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class UNet(Model):
    def __init__(self, n_filters, n_dims, name="UNet", **kwargs):
        super().__init__(name=name, **kwargs)
        # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
        self.cblock1 = EncoderBlock(
            n_filters=n_filters, dropout_prob=0, max_pooling=True
        )
        self.cblock2 = EncoderBlock(
            n_filters=n_filters * 2, dropout_prob=0, max_pooling=True
        )
        self.cblock3 = EncoderBlock(
            n_filters=n_filters * 4, dropout_prob=0, max_pooling=True
        )
        self.cblock4 = EncoderBlock(
            n_filters=n_filters * 8, dropout_prob=0.3, max_pooling=True
        )
        self.cblock5 = EncoderBlock(
            n_filters=n_filters * 16, dropout_prob=0.3, max_pooling=False
        )

        # Decoder includes multiple mini blocks with decreasing number of filters
        # Observe the skip connections from the encoder are given as input to the decoder
        # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
        self.ublock6 = DecoderBlock(n_filters=n_filters * 8)
        self.ublock7 = DecoderBlock(n_filters=n_filters * 4)
        self.ublock8 = DecoderBlock(n_filters=n_filters * 2)
        self.ublock9 = DecoderBlock(n_filters=n_filters)

        self.conv1 = layers.Conv2D(
            n_filters,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        self.conv2 = layers.Conv2D(n_dims, 1, padding="same")

    def call(self, inputs):
        ## Encoder Path
        x, z1 = self.cblock1(inputs)
        x, z2 = self.cblock2(x)
        x, z3 = self.cblock3(x)
        x, z4 = self.cblock4(x)
        x, _ = self.cblock5(x)

        ## Decoder Path
        x = self.ublock6([x, z4])
        x = self.ublock7([x, z3])
        x = self.ublock8([x, z2])
        x = self.ublock9([x, z1])

        x = self.conv1(x)
        x = self.conv2(x)

        return x


"""Functional generation of UNet, good for validation of above.
"""


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    Dropout can be added for regularization to prevent overfitting.
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
    # Proper insitialization prevents exploding and vanishing gradients
    # 'Same' padding will pad input to conv layer so output is same height and width.
    conv = layers.Conv2D(
        n_filters,
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(inputs)
    conv = layers.Conv2D(
        n_filters,
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(conv)

    # Batch Normalization will normalize the output of the last layer based on the batch's mean and std dev
    conv = layers.BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:
        conv = layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection


def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = layers.Conv2DTranspose(
        n_filters,
        (3, 3),  # Kernel size
        strides=(2, 2),
        padding="same",
    )(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = layers.concatenate([up, skip_layer_input], axis=3)

    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = layers.Conv2D(
        n_filters,
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(merge)
    conv = layers.Conv2D(
        n_filters,
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(conv)
    return conv


def UNetGen(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """

    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = layers.Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(
        cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True
    )
    cblock3 = EncoderMiniBlock(
        cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=True
    )
    cblock4 = EncoderMiniBlock(
        cblock3[0], n_filters * 8, dropout_prob=0.3, max_pooling=True
    )
    cblock5 = EncoderMiniBlock(
        cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False
    )

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1], n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    x = layers.Conv2D(
        n_filters, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(ublock9)

    x = layers.Conv2D(n_classes, 1, padding="same")(x)

    # Define the model
    model = Model(inputs=inputs, outputs=x, name="U-Net")

    return model
