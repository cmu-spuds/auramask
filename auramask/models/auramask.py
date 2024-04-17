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
from keras.losses import cosine_similarity, Loss
from keras_unet_collection import models
# import keras.ops as np


class AuraMask(Model):
    def __init__(
        self,
        n_filters,
        n_dims,
        eps=0.02,
        depth=5,
        colorspace: tuple[Callable, Callable] = None,
        name="AuraMask",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.eps = eps
        self.F = None
        self._custom_losses = []

        self.colorspace = colorspace

        self.masked = True

        self.inscale = Rescaling(2, offset=-1)

        filters = [n_filters * pow(2, i) for i in range(depth)]

        self.model = models.unet_2d(
            (None, None, 3),
            filters,
            n_labels=n_dims,
            stack_num_down=2,
            stack_num_up=2,
            activation="ReLU",
            output_activation=None,
            batch_norm=True,
            pool="max",
            unpool="nearest",
        )

    def call(self, inputs, training=False):
        if not training:
            inputs = self.colorspace[0](inputs)
        mask = self.inscale(inputs)  # Scale to -1 to 1
        mask = self.model(mask)
        if self.masked:  # Generate a mask added to the input
            mask = tanh(mask)
            mask = tf.multiply(self.eps, mask)
            out = tf.add(mask, inputs)
            out = tf.clip_by_value(out, 0.0, 1.0)
        else:  # Regenerate the input image
            out = sigmoid(mask)

        if not training:
            out = self.colorspace[1](out)

        return out, mask

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        loss_convert=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        pss_evaluation_shards=0,
        **kwargs,
    ):
        if isinstance(loss, list):
            weighted = isinstance(loss_weights, list)
            conversion = isinstance(loss_convert, list)
            w = 1.0
            c = tf.constant(False, dtype=tf.bool)
            for _ in range(len(loss)):
                loss_i = loss.pop()
                if weighted:
                    w = loss_weights.pop()
                if conversion:
                    c = tf.constant(loss_convert.pop(), dtype=tf.bool)
                if isinstance(loss_i, (EmbeddingDistanceLoss)):
                    self.F = []
                    for model in loss_i.F:
                        self.F.append((model, Mean(name=model.name), w, c))
                elif isinstance(loss_i, Loss):
                    if w > 0:
                        self._custom_losses.append((loss_i, Mean(name=loss_i.name), w, c))
                    else:
                        del loss_i
                else:
                    loss.append(loss_i)
                    if weighted:
                        loss_weights.append(w)

        return super().compile(
            optimizer,
            loss,
            metrics,
            loss_weights,
            weighted_metrics,
            run_eagerly,
            steps_per_execution,
            jit_compile,
            pss_evaluation_shards,
            **kwargs,
        )

    @tf.function
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        del x
        del sample_weight
        tloss = tf.constant(0, dtype=tf.float32)                            # tracked total loss
        y_rgb = tf.stop_gradient(self.colorspace[1](y))                     # computed rgb representation
        y_pred_rgb = tf.grad_pass_through(self.colorspace[1])(y_pred)       # computed rgb representation (with gradient passthrough i.e., identity in backward pass)

        changed_fn = lambda: (y_rgb, y_pred_rgb)
        unproc_fn = lambda: (y, y_pred)

        if self.F:
            embed_loss = tf.constant(0, dtype=tf.float32)
            for model, metric, e_w, e_c in self.F:
                tmp_y, tmp_pred = tf.cond(e_c,
                                          changed_fn,
                                          unproc_fn)
                embed_y = tf.stop_gradient(model(tmp_y, training=False))
                embed_pred = model(tmp_pred, training=False)
                sim = tf.negative(
                    cosine_similarity(
                        y_true=embed_y, y_pred=embed_pred, axis=-1
                    )
                )
                sim = tf.reduce_mean(sim)
                metric.update_state(sim)
                embed_loss = tf.add(embed_loss, sim)
            embed_loss = tf.divide(embed_loss, len(self.F))
            tloss = tf.add(tloss, tf.multiply(embed_loss, e_w))

        for model, metric, c_w, c_c in self._custom_losses:
            tmp_y, tmp_pred = tf.cond(c_c,
                            changed_fn,
                            unproc_fn)
            sim_loss = model(tmp_y, tmp_pred)
            metric.update_state(sim_loss)
            tloss = tf.add(tloss, tf.multiply(sim_loss, c_w))

        return tloss

    def get_metrics_result(self):
        all_metrics = {}
        for _, metric, _, _ in self._custom_losses:
            all_metrics[metric.name] = metric.result()
            metric.reset_state()
        if self.F:
            for _, metric, _, _ in self.F:
                all_metrics[metric.name] = metric.result()
                metric.reset_state()

        return all_metrics

    @tf.function
    def train_step(self, data):
        X, y = data

        with tf.GradientTape() as tape:
            X = self.colorspace[0](X)
            y = tf.stop_gradient(self.colorspace[0](y))
            y_pred, _ = self(X, training=True)  # Forward pass
            loss = self.compute_loss(y=y, y_pred=y_pred)  # Compute loss

        # Compute Gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update Weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (including the one that tracks loss)
        # Return a dict mapping metric names to current value
        metrics = self.get_metrics_result()
        metrics["loss"] = loss
        return metrics

    def test_step(self, data):
        X, y = data

        y_pred, _ = self(X, training=False)

        y = self.colorspace[0](y)
        y_pred = self.colorspace[0](y_pred)

        # Updates stateful loss metrics.
        loss = self.compute_loss(y=y, y_pred=y_pred)
        metrics = self.get_metrics_result()
        metrics["loss"] = loss
        return metrics
