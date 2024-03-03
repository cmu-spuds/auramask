import tensorflow as tf
from keras import Model
from auramask.losses.embeddistance import EmbeddingDistanceLoss
from auramask.losses.perceptual import PerceptualLoss
from keras.metrics import Mean
from keras.losses import cosine_similarity
from keras_unet_collection import models, base, utils
# import keras.ops as np

class AuraMask(Model):
    def __init__(self,
                 n_filters,
                 n_dims,
                 eps = 0.02,
                 depth=5,
                 name="AuraMask",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
        self.F = None
        self.Lpips = None
        
        filters = [n_filters * pow(2, i) for i in range(depth)]
        
        self.model = models.unet_2d((None, None, 3), filters, n_labels=n_dims,
                            stack_num_down=1, stack_num_up=1,
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool='max', unpool='nearest', weights=None)
        
    def call(self, inputs):
        x = self.model(inputs)
        
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
                    else:
                        del l
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
    
    def test_step(self, data):
        X, y, = data

        y_pred, _ = self(X, training=False)
        # Updates stateful loss metrics.
        loss = self.compute_loss(y=y, y_pred=y_pred)
        metrics = self.get_metrics_result()
        metrics['loss'] = loss
        return metrics