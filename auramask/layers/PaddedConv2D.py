from keras import layers


class PaddedConv2D(layers.Layer):
    def __init__(self, channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.padding2d = layers.ZeroPadding2D((padding, padding))
        self.conv2d = layers.Conv2D(channels, kernel_size, strides=(stride, stride))

    def call(self, x):
        x = self.padding2d(x)
        return self.conv2d(x)
