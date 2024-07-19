from typing import Any, Callable
from keras import ops, backend, Model, Loss, metrics as m

from auramask.losses.embeddistance import FaceEmbeddingLoss
from auramask.losses.zero_dce import IlluminationSmoothnessLoss
from auramask.metrics.embeddistance import PercentageOverThreshold


class AuraMask(Model):
    def __init__(
        self,
        backbone: Model,
        colorspace: tuple[Callable, Callable] = None,
        name="AuraMask",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._custom_losses = []
        self.colorspace = colorspace
        self.model = backbone

    def call(self, inputs, training=False):
        if not training:
            inputs = self.colorspace[0](inputs)

        out, mask = self.model(inputs)

        if not training:
            out = self.colorspace[1](out)

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
        x_rgb, x_mod = x
        y_pred, mask = y_pred
        tloss = 0  # tracked total loss
        y_pred_rgb = self.colorspace[1](
            y_pred
        )  # computed rgb representation (with gradient passthrough only for hsv_to_rgb

        idx = 0

        for loss, metric, l_w, l_c in self._custom_losses:
            if isinstance(loss, FaceEmbeddingLoss):
                tmp_y, tmp_pred = (
                    (y[0][idx], y_pred_rgb) if l_c is True else (y[0][idx], y_pred)
                )
                idx += 1
            elif isinstance(loss, IlluminationSmoothnessLoss):
                tmp_pred = mask
                tmp_y = None
            else:
                tmp_y, tmp_pred = (
                    (x_rgb, y_pred_rgb) if l_c is True else (x_mod, y_pred)
                )
            sim_loss = loss(tmp_y, tmp_pred)
            metric.update_state(sim_loss)
            tloss = ops.add(tloss, ops.multiply(sim_loss, l_w))

        return tloss

    # def save(self, filepath, overwrite=True, **kwargs):
    #     return self.model.save(filepath, overwrite, **kwargs)

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
            data  # X is input image data, y is pre-computed set of embeddings ((N Embeddings), (N Names))
        )

        X_mod = ops.copy(X)

        with GradientTape() as tape:
            X_mod = self.colorspace[0](X_mod)  # Convert to chosen colorspace
            y_pred, mask = self(X_mod, training=True)  # Forward pass with
            loss = self.compute_loss(
                x=(X, X_mod), y=y, y_pred=(y_pred, mask)
            )  # Compute loss

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

        y_pred, mask = self(X, training=False)

        y_pred = self.colorspace[0](y_pred)

        X_mod = ops.copy(X)

        X_mod = self.colorspace[0](X)  # Convert to chosen colorspace

        # Updates stateful loss metrics.
        loss = self.compute_loss(x=(X, X_mod), y=y, y_pred=(y_pred, mask))
        metrics = self.compute_metrics(x=X, y=y, y_pred=y_pred, sample_weight=None)
        metrics["loss"] = loss
        return metrics
