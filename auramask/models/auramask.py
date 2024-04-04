from types import NoneType
from typing import Callable
import tensorflow as tf
from keras import Model
from auramask.losses.embeddistance import EmbeddingDistanceLoss
from auramask.losses.perceptual import PerceptualLoss
from auramask.losses.aesthetic import AestheticLoss
from auramask.losses.ssim import SSIMLoss
from keras.activations import tanh, sigmoid
from keras.layers import Rescaling
from keras.metrics import Mean
from keras.losses import cosine_similarity, MeanSquaredError
from keras_unet_collection import models
# import keras.ops as np

class AuraMask(Model):
    def __init__(self,
                 n_filters,
                 n_dims,
                 eps = 0.02,
                 depth=5,
                 colorspace: tuple[Callable, Callable] | NoneType=None,
                 name="AuraMask",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
        self.F = None
        self.Lpips = None
        self.A = None

        self.colorspace = colorspace

        self.masked = True

        self.inscale = Rescaling(2, offset=-1)
        
        filters = [n_filters * pow(2, i) for i in range(depth)]
        
        self.model = models.unet_2d((None, None, 3), filters, n_labels=n_dims,
                            stack_num_down=2, stack_num_up=2,
                            activation='ReLU', output_activation=None, 
                            batch_norm=True, pool='max', unpool='nearest')

    def call(self, inputs, training=False):
        if not training and self.colorspace:
            inputs = self.colorspace[0](inputs)
        mask = self.inscale(inputs)  # Scale to -1 to 1
        mask = self.model(mask)
        if self.masked:             # Generate a mask added to the input
            mask = tanh(mask)
            mask = tf.multiply(self.eps, mask)
            out = tf.add(mask, inputs)
            out = tf.clip_by_value(out, 0., 1.)
        else:                       # Regenerate the input image
            out = sigmoid(mask)

        if not training and self.colorspace:
            out = self.colorspace[1](out)

        return out, mask

    def compile(self, optimizer="rmsprop", loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, pss_evaluation_shards=0, **kwargs):
        if isinstance(loss, list):
            weighted = isinstance(loss_weights, list)
            w = 1.
            for _ in range(len(loss)):
                l = loss.pop()
                if weighted:
                    w = loss_weights.pop()
                if isinstance(l, (PerceptualLoss, MeanSquaredError, SSIMLoss)):
                    if not self.Lpips and w>0:
                        self.Lpips = [(l, Mean(name=l.name), w)]
                    elif w>0:
                        self.Lpips.append((l, Mean(name=l.name), w))
                    else:
                        del l
                elif isinstance(l, EmbeddingDistanceLoss):
                    self.F = []
                    for model in l.F:
                        self.F.append((model, Mean(name=model.name), w))
                elif isinstance(l, AestheticLoss):
                    self.A = (l, Mean(name=l.name), w)
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
            if self.F:
                embed_loss = tf.constant(0, dtype=tf.float32)
                for model, metric, e_w in self.F:
                    with tf.name_scope(model.name):
                        embed_y = tf.stop_gradient(model(y, training=False))
                        embed_pred = model(y_pred, training=False)
                        sim = tf.negative(cosine_similarity(y_true=embed_y, y_pred=embed_pred, axis=-1))
                        sim = tf.reduce_mean(sim)
                    metric.update_state(sim)
                    embed_loss = tf.add(embed_loss, sim)
                embed_loss = tf.divide(embed_loss, len(self.F))
                tloss = tf.add(tloss, tf.multiply(embed_loss, e_w))

        with tf.name_scope("Perceptual"):
            if self.Lpips:
                for model, metric, p_w in self.Lpips:
                    sim_loss = model(y, y_pred)
                    metric.update_state(sim_loss)
                    tloss = tf.add(tloss, tf.multiply(sim_loss, p_w))

        with tf.name_scope("Aesthetic"):
            if self.A:
                model, metric, a_w = self.A
                a_loss = model(y, y_pred)
                metric.update_state(a_loss)
                tloss = tf.add(tloss, tf.multiply(a_loss, a_w))

        return tloss

    def get_metrics_result(self):
        all_metrics = {}
        if self.A:
            _, metric, _ = self.A
            all_metrics[metric.name] = metric.result()
            metric.reset_state()
        if self.Lpips:
            for _, metric, _ in self.Lpips:
                all_metrics[metric.name] = metric.result()
                metric.reset_state()
        if self.F:
            for _, metric, _ in self.F:
                all_metrics[metric.name] = metric.result()
                metric.reset_state()

        return all_metrics

    @tf.function
    def train_step(self, data):
        X, y = data

        with tf.GradientTape() as tape:
            if self.colorspace:                                                 # If a different colorspace provided, convert to it
                X = self.colorspace[0](X)
                y = tf.stop_gradient(self.colorspace[0](y))
            y_pred, _ = self(X, training=True)                                  # Forward pass
            loss = self.compute_loss(y=y, y_pred=y_pred)                        # Compute loss

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
        X, y = data

        y_pred, _ = self(X, training=False)

        if self.colorspace:
            y = self.colorspace[0](y)

        # Updates stateful loss metrics.
        loss = self.compute_loss(y=y, y_pred=y_pred)
        metrics = self.get_metrics_result()
        metrics['loss'] = loss
        return metrics