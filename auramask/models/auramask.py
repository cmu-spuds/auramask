from typing import Any, Callable, Optional, Tuple, List, Dict
from keras import ops, backend, Model, metrics as m, KerasTensor, Loss
from auramask.losses.embeddistance import FaceEmbeddingLoss
from auramask.losses.zero_dce import IlluminationSmoothnessLoss


class AuraMask(Model):
    def __init__(
        self,
        *args,
        **kwargs,
        # colorspace: tuple[Callable, Callable] = None,
    ):
        super().__init__(*args, **kwargs)
        self._my_losses = []
        self._gradient_alteration = None
        self._loss_weights = None
        # self.colorspace = colorspace

    @property
    def losses(self) -> List[Loss]:
        return self._my_losses

    @property
    def metrics(self) -> List[m.Metric]:
        return self._metrics + self._loss_trackers

    @property
    def loss_weights(self) -> List[KerasTensor]:
        return list(self._loss_weights)

    @loss_weights.setter
    def loss_weights(self, value):
        assert len(value) == len(self._loss_weights)
        self._loss_weights = ops.convert_to_tensor(value)

    def get_loss_bundle(self) -> Tuple[List[Loss], List[KerasTensor], List[m.Metric]]:
        return (self._losses, self._loss_weights, self._loss_trackers)

    def call(
        self, inputs: KerasTensor, training: bool = False
    ) -> Tuple[KerasTensor, KerasTensor]:
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
        gradient_alter: Callable | None = None,
        metrics: Any | None = None,
        weighted_metrics: Any | None = None,
        run_eagerly: bool = False,
        steps_per_execution: int = 1,
        jit_compile: str = "auto",
        auto_scale_loss: bool = True,
    ):
        super().compile(
            optimizer=optimizer,
            loss=None,
            loss_weights=None,
            metrics=None,
            weighted_metrics=None,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            auto_scale_loss=auto_scale_loss,
        )
        self._my_losses = loss
        self._loss_weights = ops.convert_to_tensor(loss_weights)
        self._loss_trackers = [m.Mean(name=loss_i.name) for loss_i in loss]
        self._metrics = metrics if metrics else []
        self._gradient_alteration = gradient_alter

    def compute_loss(
        self,
        x: KerasTensor,
        y: KerasTensor,
        y_pred: KerasTensor,
        sample_weight: Optional[KerasTensor] = None,
    ) -> List[KerasTensor]:
        del sample_weight
        # Predictions are passed as tuple (y_pred, mask)
        y_pred, mask = y_pred
        losses = [0.0]

        for i, loss in enumerate(self._my_losses):
            weight = self.loss_weights[i]
            metric = self._loss_trackers[i]
            tmp_loss = 0.0
            if isinstance(loss, FaceEmbeddingLoss):
                tmp_loss = loss(y_true=ops.stop_gradient(x), y_pred=y_pred)
                losses[0] = ops.add(losses[0], ops.multiply(tmp_loss, weight))
            elif isinstance(loss, IlluminationSmoothnessLoss):
                tmp_loss = loss(y_true=y, y_pred=mask)
                losses.append(ops.multiply(tmp_loss, weight))
            else:
                tmp_loss = loss(y_true=y, y_pred=y_pred)
                losses.append(ops.multiply(tmp_loss, weight))
            metric.update_state(ops.stop_gradient(tmp_loss))

        return losses

    # def save(self, filepath: str, overwrite: bool=True, **kwargs):
    #     config: dict = self.get_config()
    #     weights: list = self.get_weights()
    #     model: Model = Model.from_config(config)
    #     model.load_weights = weights
    #     return self.model.save(filepath, overwrite, **kwargs)

    def compute_metrics(
        self,
        x: KerasTensor,
        y: KerasTensor,
        y_pred: KerasTensor,
        sample_weight: Optional[KerasTensor] = None,
    ) -> Dict[str, KerasTensor]:
        del sample_weight

        y_pred, _ = y_pred

        for metric in self._metrics:
            metric.update_state(y, y_pred)
        return self.get_metrics_result()

    def get_metrics_result(self) -> Dict[str, KerasTensor]:
        all_metrics: dict = {}
        for metric in self.metrics:
            all_metrics[metric.name] = metric.result()
            metric.reset_state()
        return all_metrics

    def train_step(self, *args, **kwargs) -> Dict[str, KerasTensor]:
        if backend.backend() == "jax":
            return self._jax_train_step(*args, **kwargs)
        elif backend.backend() == "tensorflow":
            return self._tensorflow_train_step(*args, **kwargs)
        elif backend.backend() == "torch":
            return self._torch_train_step(*args, **kwargs)

    def _tensorflow_train_step(self, data) -> Dict[str, KerasTensor]:
        from tensorflow import GradientTape

        X, y = data  # X is input image data, y is the target image

        with GradientTape() as tape:
            y_pred = self(X, training=True)  # Forward pass with
            loss = self.compute_loss(x=X, y=y, y_pred=y_pred)  # Compute loss
            loss = ops.sum(loss)
            scaled_loss = self.optimizer.scale_loss(loss)

        # Compute Gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)

        # Update Weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (including the one that tracks loss)
        # Return a dict mapping metric names to current value
        metrics = self.compute_metrics(x=X, y=y, y_pred=y_pred, sample_weight=None)
        metrics["loss"] = loss
        return metrics

    def __compute_gradient(
        self, loss: list
    ) -> Tuple[List[KerasTensor], List[KerasTensor]]:
        if self._gradient_alteration is None:
            loss = ops.sum(loss)
            scaled_loss = self.optimizer.scale_loss(loss)
            scaled_loss.backward()
            trainable_weights = [v for v in self.trainable_weights]
            gradients = [v.value.grad for v in trainable_weights]
        else:
            scaled_loss = [self.optimizer.scale_loss(ls) for ls in loss]
            gradients, trainable_weights = self._gradient_alteration(
                scaled_loss, self.trainable_weights
            )
        return gradients, trainable_weights

    def _torch_train_step(self, data) -> Dict[str, KerasTensor]:
        import torch

        X, y = data  # X is input image data, y is target image
        X = ops.convert_to_tensor(X)
        y = ops.convert_to_tensor(y)
        self.zero_grad()

        y_pred = self(X, training=True)
        loss = self.compute_loss(x=X, y=y, y_pred=y_pred)

        gradients, trainable_weights = self.__compute_gradient(loss)

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
            metrics = self.compute_metrics(x=X, y=y, y_pred=y_pred, sample_weight=None)
            metrics["loss"] = ops.sum(loss)
        return metrics

    def test_step(self, *args, **kwargs):
        if backend.backend() == "jax":
            return self._jax_test_step(*args, **kwargs)
        elif backend.backend() == "tensorflow":
            return self._tensorflow_test_step(*args, **kwargs)
        elif backend.backend() == "torch":
            return self._torch_test_step(*args, **kwargs)

    def _torch_test_step(self, data) -> Dict[str, KerasTensor]:
        import torch

        X, y = data  # X is input image data, y is target image

        X = ops.convert_to_tensor(X)
        y = ops.convert_to_tensor(y)

        with torch.no_grad():
            y_pred = self(X, training=False)

            # Updates stateful loss metrics.
            loss = self.compute_loss(x=X, y=y, y_pred=y_pred)
            loss = ops.sum(loss)
            metrics = self.compute_metrics(x=X, y=y, y_pred=y_pred, sample_weight=None)
        metrics["loss"] = loss
        return metrics
