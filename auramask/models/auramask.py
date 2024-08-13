from typing import Any
from keras import ops, backend, Model, Loss, metrics as m
from auramask.losses.embeddistance import FaceEmbeddingLoss
from auramask.losses.zero_dce import IlluminationSmoothnessLoss
from auramask.metrics.embeddistance import PercentageOverThreshold


class AuraMask(Model):
    def __init__(
        self,
        *args,
        **kwargs,
        # colorspace: tuple[Callable, Callable] = None,
    ):
        self._custom_losses = []
        # self.colorspace = colorspace
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=False):
        # if not training:
        #     inputs = self.colorspace[0](inputs)

        out, mask = super().call(inputs, training)

        # if not training:
        #     out = self.colorspace[1](out)

        return out, mask

    def compile(
        self,
        optimizer: str = "rmsprop",
        loss: Any | None = None,
        loss_weights: Any | None = None,
        loss_convert=None,
        metrics: Any | None = None,
        weighted_metrics: Any | None = None,
        run_eagerly: bool = False,
        steps_per_execution: int = 1,
        jit_compile: str = "auto",
        auto_scale_loss: bool = True,
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
                    self._custom_losses.append((loss_i, m.Mean(name=loss_i.name), w, c))
                else:
                    loss.append(loss_i)
                    if weighted:
                        loss_weights.append(w)

        return super().compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            auto_scale_loss=auto_scale_loss,
        )

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        del sample_weight
        y_pred, mask = y_pred
        tloss = 0  # tracked total loss

        idx = 0

        for loss, metric, l_w, l_c in self._custom_losses:
            if isinstance(loss, FaceEmbeddingLoss):
                tmp_y, tmp_pred = (y[idx], y_pred)
                idx += 1
            elif isinstance(loss, IlluminationSmoothnessLoss):
                tmp_pred = mask
                tmp_y = None
            else:
                tmp_y, tmp_pred = (
                    (x, y_pred)
                    # (x_rgb, y_pred_rgb) if l_c is True else (x_mod, y_pred)
                )
            sim_loss = loss(tmp_y, tmp_pred)
            metric.update_state(sim_loss)
            tloss = ops.add(tloss, ops.multiply(sim_loss, l_w))

        return tloss

    # def save(self, filepath, overwrite=True, **kwargs):
    #     return self.model.save(filepath, overwrite, **kwargs)

    def compute_metrics(self, x, y, y_pred, sample_weight):
        del sample_weight

        y_pred, mask = y_pred
        idx = 0

        for metric in self._metrics:
            if isinstance(metric, PercentageOverThreshold):
                metric.update_state(y[idx], y_pred)
                idx += 1
            else:
                metric.update_state(x, y_pred)
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

    def train_step(self, *args, **kwargs):
        if backend.backend() == "jax":
            return self._jax_train_step(*args, **kwargs)
        elif backend.backend() == "tensorflow":
            return self._tensorflow_train_step(*args, **kwargs)
        elif backend.backend() == "torch":
            return self._torch_train_step(*args, **kwargs)

    def _tensorflow_train_step(self, data):
        from tensorflow import GradientTape

        X, y = (
            data  # X is input image data, y is pre-computed set of N embeddings (N x batch x Embeddings)
        )

        X_mod = ops.copy(X)

        with GradientTape() as tape:
            # X_mod = self.colorspace[0](X_mod)  # Convert to chosen colorspace
            y_pred = self(X_mod, training=True)  # Forward pass with
            loss = self.compute_loss(x=X_mod, y=y, y_pred=y_pred)  # Compute loss

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

    def _torch_train_step(self, data):
        import torch

        X, y = data

        X_mod = ops.copy(X)

        self.zero_grad()

        y_pred = self(X_mod, training=True)
        loss = self.compute_loss(x=X_mod, y=y, y_pred=y_pred)

        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        metrics = self.compute_metrics(x=X, y=y, y_pred=y_pred, sample_weight=None)
        metrics["loss"] = loss
        return metrics

    def test_step(self, data):
        X, y = (
            data  # X is input image data, y is pre-computed set of embeddings ((N Embeddings), (N Names))
        )

        y_pred = self(X, training=False)

        X_mod = ops.copy(X)

        # Updates stateful loss metrics.
        loss = self.compute_loss(x=X_mod, y=y, y_pred=y_pred)
        metrics = self.compute_metrics(x=X, y=y, y_pred=y_pred, sample_weight=None)
        metrics["loss"] = loss
        return metrics
