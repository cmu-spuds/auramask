import tensorflow as tf
from keras import Model
from keras.losses import Loss
from keras.metrics import Mean
from keras.layers import Layer, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, concatenate
# import keras.ops as np

class EncoderBlock(Layer):
    def __init__(self, n_filters=32, kernel=3, dropout_prob=0.3, max_pooling=True, name='UEncoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dropout_prob = dropout_prob
        self.max_pooling = max_pooling

        self.conv1 = Conv2D(n_filters,
                            kernel,
                            activation='relu',
                            padding='same',
                            kernel_initializer='HeNormal')
        self.conv2 = Conv2D(n_filters,
                            kernel,
                            activation='relu',
                            padding='same',
                            kernel_initializer='HeNormal')

        self.bn = BatchNormalization()
        self.do = Dropout(dropout_prob) if dropout_prob > 0 else None
        self.mp = MaxPooling2D(pool_size=(2,2)) if max_pooling else None

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Batch norm to normalize output of last layer based on batch mean and std
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

class DecoderBlock(Layer):
    def __init__(self, kernel=3, stride=2, n_filters=32, name="UDecoder", **kwargs):
        super().__init__(name=name, **kwargs)

        self.up1 = Conv2DTranspose(
                             n_filters,
                             (kernel,kernel),    # Kernel size
                             strides=(stride,stride),
                             padding='same')

        self.conv1 = Conv2D(n_filters,
                            kernel,
                            activation='relu',
                            padding='same',
                            kernel_initializer='HeNormal')        

        self.conv2 = Conv2D(n_filters,
                            kernel,
                            activation='relu',
                            padding='same',
                            kernel_initializer='HeNormal')

    def call(self, x):
        x, skp = x
        
        # Start with a transpose convolution layer to first increase the size of the image
        x = self.up1(x)

        # Merge the skip connection from previous block to prevent information loss
        x = concatenate([x, skp], axis=3)
    
        # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
        # The parameters for the function are similar to encoder
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class AuraMask(Model):
    def __init__(self,
                 n_filters,
                 n_dims,
                 eps = 0.2,
                 name="AuraMask",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = None
        self.eps = eps

        # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
        self.cblock1 = EncoderBlock(n_filters=n_filters, dropout_prob=0, max_pooling=True)
        self.cblock2 = EncoderBlock(n_filters=n_filters*2,dropout_prob=0, max_pooling=True)
        self.cblock3 = EncoderBlock(n_filters=n_filters*4,dropout_prob=0, max_pooling=True)
        self.cblock4 = EncoderBlock(n_filters=n_filters*8,dropout_prob=0.3, max_pooling=True)
        self.cblock5 = EncoderBlock(n_filters=n_filters*16, dropout_prob=0.3, max_pooling=False)
    
        # Decoder includes multiple mini blocks with decreasing number of filters
        self.ublock6 = DecoderBlock(n_filters=n_filters * 8)
        self.ublock7 = DecoderBlock(n_filters=n_filters * 4)
        self.ublock8 = DecoderBlock(n_filters=n_filters * 2)
        self.ublock9 = DecoderBlock(n_filters=n_filters)
        
        self.conv1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(n_dims, 1, padding='same')

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
        
        x = tf.tanh(x)
        x = tf.multiply(self.eps, x)
        x = tf.add(x, inputs)
        x = tf.clip_by_value(x, 0., 1.)

        return x

    def compile(self, optimizer="rmsprop", loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, pss_evaluation_shards=0, **kwargs):
        self.loss_fn = loss
        
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, pss_evaluation_shards, **kwargs)

    # def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
    #     for loss in self.losses + self.tracked_metrics:
    #         if metric.name == "loss":
    #             metric.update_state(loss)
    #         else:
    #             metric.update_state(x, x_adv)
    #     return super().compute_loss(x, y, y_pred, sample_weight)

    # def train_step(self, data):
    #     x_adv, x = data
        
    #     with tf.GradientTape() as tape:
    #         x_adv = self(x, training=True) # Forward pass
    #         # Compute Loss configured in 'compile()'
    #         loss = self.compute_loss(y=x, y_pred=x_adv)
    #     # Compute Gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     # Update Weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     # Update metrics (including the one that tracks loss)
    #     return self.compute_metrics(x=x, y=x, y_pred=x_adv, sample_weight=None)