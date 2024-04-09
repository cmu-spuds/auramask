import io
import os
from typing import Dict

import wandb
from keras import Model
from wandb.keras import WandbEvalCallback


def get_model_summary(model):
    stream = io.StringIO()
    model.summary(
        print_fn=lambda x: stream.write(x + "\n"),
        expand_nested=True,
        show_trainable=True,
    )
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


class ImageCallback(WandbEvalCallback):

    def __init__(
        self,
        validation_data,
        data_table_columns,
        pred_table_columns,
        num_samples=100,
    ):
        super().__init__(data_table_columns=data_table_columns, pred_table_columns=pred_table_columns)
        self.x, self.y = validation_data

    def add_ground_truth(self, logs: Dict[str, float] | None = None) -> None:
        for idx, (orig, aug) in enumerate(zip(self.x, self.y)):
            self.data_table.add_data(idx, wandb.Image(orig), wandb.Image(aug))

    def add_model_predictions(self, epoch: int, logs: Dict[str, float] | None = None) -> None:
        y, mask = self.model(self.x, training=False)
        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = y[idx]
            m = mask[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                wandb.Image(pred),
                wandb.Image(m)
            )