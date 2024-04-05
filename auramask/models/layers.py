from keras import Sequential
from keras.layers import (
    Layer,
    Conv2D,
    SeparableConv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    Conv2DTranspose,
    concatenate,
    Add,
)


class ResBlock(Layer):
    def __init__(
        self,
        n_filters=64,
        kernel=3,
        padding="same",
        activation="relu",
        name="resblock",
        kernel_initializer="HeNormal",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.conv1 = Conv2D(
            n_filters,
            kernel,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer,
        )
        self.conv2 = Conv2D(
            n_filters,
            kernel,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer,
        )
        self.ladd = Add()

    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.ladd([input, x])
        return x


class EncoderBlock(Layer):
    def __init__(
        self,
        n_filters=32,
        kernel=3,
        dropout_prob=0.3,
        max_pooling=True,
        n_layers=1,
        name="UEncoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.dropout_prob = dropout_prob
        self.max_pooling = max_pooling

        self.convs = Sequential(name="convs")

        self._build_conv_layers(n_layers, n_filters, kernel)

        self.bn = BatchNormalization()
        self.do = Dropout(dropout_prob)
        self.mp = MaxPooling2D(pool_size=(2, 2)) if max_pooling else None

    def _build_conv_layers(self, n_layers, n_filters, kernel):
        for _ in range(n_layers):
            self.convs.add(
                SeparableConv2D(
                    n_filters,
                    kernel,
                    activation="relu",
                    padding="same",
                    kernel_initializer="HeNormal",
                )
            )

    def call(self, inputs, training):
        x = self.convs(inputs)

        # Batch norm to normalize output of last layer based on batch mean and std
        x = self.bn(x, training=training)

        # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
        x = self.do(x)

        # Pooling reduces the size of the image while keeping the number of channels same
        # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
        # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
        if self.mp:
            next_layer = self.mp(x)
        else:
            next_layer = x

        return next_layer, x


class ResEncoderBlock(EncoderBlock):
    def __init__(
        self,
        n_filters=32,
        kernel=3,
        dropout_prob=0.3,
        max_pooling=True,
        n_layers=2,
        name="ResEncoder",
        **kwargs,
    ):
        super().__init__(
            None, kernel, dropout_prob, max_pooling, n_layers, name, **kwargs
        )

    def _build_conv_layers(self, n_layers, n_filters, kernel):
        for _ in range(n_layers):
            self.convs.add(
                ResBlock(
                    n_filters,
                    kernel,
                    activation="relu",
                    padding="same",
                    kernel_initializer="HeNormal",
                )
            )


class DecoderBlock(Layer):
    def __init__(
        self, kernel=3, stride=2, n_filters=32, n_layers=1, name="UDecoder", **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.up1 = Conv2DTranspose(
            n_filters,
            (kernel, kernel),  # Kernel size
            strides=(stride, stride),
            padding="same",
        )

        self.convs = Sequential(name="convs")

        self._build_conv_layers(n_layers, n_filters, kernel)

    def _build_conv_layers(self, n_layers, n_filters, kernel):
        for _ in range(n_layers):
            self.convs.add(
                Conv2D(
                    n_filters,
                    kernel,
                    activation="relu",
                    padding="same",
                    kernel_initializer="HeNormal",
                )
            )

    def call(self, input):
        x, skp = input

        # Start with a transpose convolution layer to first increase the size of the image
        x = self.up1(x)

        # Merge the skip connection from previous block to prevent information loss
        x = concatenate([x, skp], axis=3)

        # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
        # The parameters for the function are similar to encoder
        x = self.convs(x)

        return x
