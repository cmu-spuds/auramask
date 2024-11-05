import numpy as np
import wandb
from keras import callbacks


class AuramaskStopOnNaN(callbacks.TerminateOnNaN):
    def __init__(self):
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                wandb.alert(
                    title="NaN Termination",
                    text="NaN value detected in one or more of the losses",
                    level=wandb.AlertLevel.ERROR,
                )
                self.model.stop_training = True
