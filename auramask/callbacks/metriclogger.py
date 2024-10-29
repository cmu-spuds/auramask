from typing import Any, Dict, Optional
import wandb
from keras import ops
from wandb.integration.keras import (
    WandbMetricsLogger,
)


class AuramaskWandbMetrics(WandbMetricsLogger):
    def _get_lr(self) -> float | None:
        if ops.is_tensor(
            self.model.optimizer.learning_rate,
        ) or (
            hasattr(self.model.optimizer.learning_rate, "shape")
            and self.model.optimizer.learning_rate.shape == ()
        ):
            return ops.convert_to_numpy(self.model.optimizer.learning_rate)
        try:
            return ops.convert_to_numpy(
                self.model.optimizer.learning_rate(step=self.global_step)
            )
        except Exception as e:
            wandb.termerror(f"Unable to log learning rate: {e}", repeat=False)
            return None

    def _get_loss_weights(self) -> dict[str, float] | None:
        if hasattr(self.model, "loss_weights") and hasattr(self.model, "losses"):
            weights = {}
            for loss, weight in zip(self.model.losses, self.model.loss_weights):
                weights["epoch/" + loss.name + "_weight"] = ops.convert_to_numpy(weight)
            return weights
        else:
            return None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch."""
        logs = (
            dict()
            if logs is None
            else {
                f"epoch/{k}": ops.convert_to_numpy(ops.cast(v, "float32"))
                for k, v in logs.items()
            }
        )

        logs["epoch/epoch"] = epoch

        lr = self._get_lr()
        if lr is not None:
            logs["epoch/learning_rate"] = lr

        weights = self._get_loss_weights()

        if weights is not None:
            logs.update(weights)

        wandb.log(logs)
