from typing import Dict, Optional, Any

import wandb
from wandb.integration.keras import (
    WandbEvalCallback,
)
from wandb.integration.keras.callbacks.model_checkpoint import SaveStrategy
from keras import preprocessing, ops


class AuramaskCallback(WandbEvalCallback):
    def __init__(
        self,
        validation_data,
        data_table_columns,
        pred_table_columns,
        num_samples=100,
        log_freq=1,
    ):
        super().__init__(
            data_table_columns=data_table_columns, pred_table_columns=pred_table_columns
        )
        self.x = validation_data[:num_samples]
        self.log_freq = log_freq

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if wandb.run.step > 1:
            pass
        else:
            wandb.log(
                {
                    "image": [
                        wandb.Image(
                            preprocessing.image.array_to_img(x_i * 255, scale=False)
                        )
                        for x_i in self.x
                    ],
                },
                step=0,
            )

    def on_epoch_end(
        self, epoch: int, logs: Dict[SaveStrategy, float] | None = None
    ) -> None:
        if epoch % self.log_freq == 0:
            self.save_results()

    def save_results(self):
        y, mask = self.model(self.x, training=False)
        if ops.shape(mask)[-1] > 3 and ops.shape(mask)[-1] % 3 == 0:
            data = {}
            r = ops.split(mask, 8, axis=-1)
            y = ops.convert_to_numpy(y)
            r = ops.convert_to_numpy(r)

            for i, r_n in enumerate(r):
                data["r%d" % i] = [
                    wandb.Image(
                        preprocessing.image.array_to_img(m_i * 255, scale=False)
                    )
                    for m_i in r_n
                ]

            data["image"] = [
                wandb.Image(preprocessing.image.array_to_img(y_i * 255, scale=False))
                for y_i in y
            ]
            wandb.log(data, step=wandb.run.step)

        else:
            y = ops.convert_to_numpy(y)
            mask = ops.convert_to_numpy(mask)
            wandb.log(
                {
                    "image": [
                        wandb.Image(
                            preprocessing.image.array_to_img(y_i * 255, scale=False)
                        )
                        for y_i in y
                    ],
                    "mask": [
                        wandb.Image(
                            preprocessing.image.array_to_img(m_i * 255, scale=False)
                        )
                        for m_i in mask
                    ],
                },
                step=wandb.run.step,
            )

    def add_ground_truth(self, logs: Dict[str, float] | None = None) -> None:
        pass

    def add_model_predictions(
        self, epoch: int, logs: Dict[str, float] | None = None
    ) -> None:
        pass

    def on_train_end(self, logs: Dict[str, float] | None = None) -> None:
        self.save_results()
