from typing import Callable
import tensorflow as tf
from keras import Model, layers
from keras.activations import tanh
from keras.metrics import Mean
from keras.losses import Loss
from keras_unet_collection import models

from auramask.losses.embeddistance import FaceEmbeddingLoss
from auramask.metrics.embeddistance import PercentageOverThreshold
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
        self._custom_losses = []

        self.scaling = layers.Rescaling(scale=2, offset=-1)

        self.colorspace = colorspace

        self.embed_order = []

        filters = [n_filters * pow(2, i) for i in range(depth)]

        self.model: Model = models.unet_2d(
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

        # self.model = models.r2_unet_2d(
        #     (None, None, 3),
        #     filters,
        #     n_labels=n_dims,
        #     stack_num_down=2,
        #     stack_num_up=2,
        #     recur_num=2,
        #     activation="GELU",
        #     output_activation=None,
        #     batch_norm=True,
        #     pool="max",
        #     unpool="nearest",
        #     name="r2unet",
        # )

    def call(self, inputs, training=False):
        if not training:
            inputs = self.colorspace[0](inputs)

        mask = self.scaling(inputs)
        mask = self.model(mask)
        mask = tanh(mask)
        mask = tf.multiply(self.eps, mask)
        out = tf.add(mask, inputs)
        out = tf.clip_by_value(out, 0.0, 1.0)

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
        self._metrics = metrics
        if isinstance(loss, list):
            weighted = isinstance(loss_weights, list)
            conversion = isinstance(loss_convert, list)
            w = 1.0
            c = False
            for _ in range(len(loss)):
                loss_i = loss.pop(0)
                if weighted:
                    w = loss_weights.pop(0)
                if conversion:
                    c = loss_convert.pop(0)
                if isinstance(loss_i, Loss):
                    if w > 0:
                        self._custom_losses.append(
                            (loss_i, Mean(name=loss_i.name), w, c)
                        )
                    else:
                        del loss_i
                else:
                    loss.append(loss_i)
                    if weighted:
                        loss_weights.append(w)

        return super().compile(
            optimizer,
            loss,
            None,
            loss_weights,
            weighted_metrics,
            run_eagerly,
            steps_per_execution,
            jit_compile,
            pss_evaluation_shards,
            **kwargs,
        )

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        del sample_weight
        x_rgb, x_mod = x
        tloss = tf.constant(0, dtype=tf.float32)  # tracked total loss
        y_pred_rgb = self.colorspace[1](
            y_pred
        )  # computed rgb representation (with gradient passthrough only for hsv_to_rgb

        idx = 0

        for loss, metric, l_w, l_c in self._custom_losses:
            if isinstance(loss, FaceEmbeddingLoss):
                # for i in tf.range(0, y[1].shape):
                #     if tf.math.equal(y[1][i], loss.f.name):
                #         break

                # idx = tf.where(y[1] == loss.f.name).numpy()[0][0] #TODO: Breaks when running in non-eager mode
                tmp_y, tmp_pred = (
                    (y[0][idx], y_pred_rgb) if l_c is True else (y[0][idx], y_pred)
                )
                idx += 1
            else:
                tmp_y, tmp_pred = (
                    (x_rgb, y_pred_rgb) if l_c is True else (x_mod, y_pred)
                )
            sim_loss = loss(tmp_y, tmp_pred)
            metric.update_state(sim_loss)
            tloss = tf.add(tloss, tf.multiply(sim_loss, l_w))

        return tloss

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        return self.model.save(filepath, overwrite, save_format, **kwargs)

    def compute_metrics(self, x, y, y_pred, sample_weight):
        del sample_weight
        y_pred_rgb = self.colorspace[1](y_pred)

        idx = 0

        for metric in self._metrics:
            if isinstance(metric, PercentageOverThreshold):
                metric.update_state(y[0][idx], y_pred_rgb)
                idx += 1
            else:
                metric.update_state(x, y_pred_rgb)
        return self.get_metrics_result()

    def get_metrics_result(self):
        all_metrics = {}
        for _, metric, _, _ in self._custom_losses:
            all_metrics[metric.name] = metric.result()
            metric.reset_state()
        for metric in self._metrics:
            all_metrics[metric.name] = metric.result()
            metric.reset_state()
        return all_metrics

    def train_step(self, data):
        X, y = (
            data  # X is input image data, y is pre-computed set of embeddings ((N Embeddings), (N Names))
        )

        with tf.GradientTape() as tape:
            X_mod = self.colorspace[0](X)  # Convert to chosen colorspace
            y_pred, _ = self(X_mod, training=True)  # Forward pass with
            loss = self.compute_loss(x=(X, X_mod), y=y, y_pred=y_pred)  # Compute loss

        # Compute Gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update Weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (including the one that tracks loss)
        # Return a dict mapping metric names to current value
        metrics = self.compute_metrics(x=X, y=y, y_pred=y_pred, sample_weight=None)
        metrics["loss"] = loss
        return metrics

    def test_step(self, data):
        X, y = (
            data  # X is input image data, y is pre-computed set of embeddings ((N Embeddings), (N Names))
        )

        y_pred, _ = self(X, training=False)

        y_pred = self.colorspace[0](y_pred)

        X_mod = self.colorspace[0](X)  # Convert to chosen colorspace

        # Updates stateful loss metrics.
        loss = self.compute_loss(x=(X, X_mod), y=y, y_pred=y_pred)
        metrics = self.compute_metrics(x=X, y=y, y_pred=y_pred, sample_weight=None)
        metrics["loss"] = loss
        return metrics
