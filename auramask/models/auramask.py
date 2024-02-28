import tensorflow as tf
from keras import Model
from keras.layers import Layer, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, concatenate
from auramask.losses.embeddistance import EmbeddingDistanceLoss
from auramask.losses.perceptual import PerceptualLoss
from keras.metrics import Mean
from keras.losses import cosine_similarity
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
        self.do = Dropout(dropout_prob)
        self.mp = MaxPooling2D(pool_size=(2,2)) if max_pooling else None

    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.conv2(x)

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

    def call(self, input):
        x, skp = input
        
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
                 eps = 0.02,
                 name="AuraMask",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
        self.F = None
        self.Lpips = None
        
        # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
        self.cblock1 = EncoderBlock(n_filters=n_filters, dropout_prob=0, max_pooling=True, name="cblock%d"%1)
        self.cblock2 = EncoderBlock(n_filters=n_filters*2,dropout_prob=0, max_pooling=True, name="cblock%d"%2)
        self.cblock3 = EncoderBlock(n_filters=n_filters*4,dropout_prob=0, max_pooling=True, name="cblock%d"%3)
        self.cblock4 = EncoderBlock(n_filters=n_filters*8,dropout_prob=0.3, max_pooling=True, name="cblock%d"%4)
        self.cblock5 = EncoderBlock(n_filters=n_filters*16, dropout_prob=0.3, max_pooling=False, name="cblock%d"%5)
    
        # Decoder includes multiple mini blocks with decreasing number of filters
        self.ublock6 = DecoderBlock(n_filters=n_filters * 8, name="ublock%d"%6)
        self.ublock7 = DecoderBlock(n_filters=n_filters * 4, name="ublock%d"%7)
        self.ublock8 = DecoderBlock(n_filters=n_filters * 2, name="ublock%d"%8)
        self.ublock9 = DecoderBlock(n_filters=n_filters, name="ublock%d"%9)
        
        self.conv1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv1")
        self.conv2 = Conv2D(n_dims, 1, padding='same', name="conv2")

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
        mask = x
        x = tf.add(x, inputs)
        x = tf.clip_by_value(x, 0., 1.)

        return x, mask

    def compile(self, optimizer="rmsprop", loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, pss_evaluation_shards=0, **kwargs):
        if isinstance(loss, list):
            weighted = isinstance(loss_weights, list)
            w = 1.
            for _ in range(len(loss)):
                l = loss.pop()
                if weighted:
                    w = loss_weights.pop()
                if isinstance(l, PerceptualLoss):
                    if w>0:
                        self.Lpips = (l.model, Mean(name='lpips'), w)
                elif isinstance(l, EmbeddingDistanceLoss):
                    self.F = []
                    for model,reg,name in l.F_set:
                        self.F.append((model, reg, Mean(name=name), w))
                else:
                    loss.append(l)
                    if weighted:
                        loss_weights.append(w)
                
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, pss_evaluation_shards, **kwargs)

    @tf.function
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        del x
        del sample_weight
        tloss = tf.constant(0, dtype=tf.float32)
        with tf.name_scope("EmbeddingDistance"):
            embed_loss = tf.constant(0, dtype=tf.float32)
            for model, shape, metric, e_w in self.F:
                with tf.name_scope(model.name):
                    embed_y = tf.stop_gradient(model(tf.image.resize(y, shape), training=False))
                    embed_pred = model(tf.image.resize(y_pred, shape), training=False)
                    sim = tf.negative(cosine_similarity(y_true=embed_y, y_pred=embed_pred, axis=-1))
                    sim = tf.reduce_mean(sim)
                metric.update_state(sim)
                embed_loss = tf.add(embed_loss, sim)
            embed_loss = tf.divide(embed_loss, len(self.F))
        tloss = tf.add(tloss, tf.multiply(embed_loss, e_w))

        with tf.name_scope("Perceptual"):
            if self.Lpips:
                model, metric, p_w = self.Lpips
                sim_loss = tf.reduce_mean(model([y, y_pred], training=False))
                metric.update_state(sim_loss)       
                tloss = tf.add(tloss, tf.multiply(sim_loss, p_w))

        return tloss

    def get_metrics_result(self):
        all_metrics = {}
        if self.Lpips:
            for metric in [self.Lpips[1]] + [metric for _, _, metric, _ in self.F]:
                all_metrics[metric.name] = metric.result()
                metric.reset_state()
        else:
            for _, _, metric, _ in self.F:
                all_metrics[metric.name] = metric.result()
                metric.reset_state()

        return all_metrics

    @tf.function
    def train_step(self, data):
        X, y = data
        
        with tf.GradientTape() as tape:
            tape.watch(X)
            y_pred, _ = self(X, training=True) # Forward pass
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute Gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update Weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (including the one that tracks loss)
        # Return a dict mapping metric names to current value
        metrics = self.get_metrics_result()
        metrics['loss'] = loss
        return metrics