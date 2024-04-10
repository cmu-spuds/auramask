import io
import os
from typing import Dict, List

import wandb
from wandb.keras import WandbEvalCallback
import tensorflow as tf
from keras.preprocessing.image import array_to_img

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


        wandb.log({
            'image': [wandb.Image(array_to_img(y_i)) for y_i in y[:5]],
            'mask': [wandb.Image(array_to_img(m_i)) for m_i in mask[:5]]
        })

        for idx in table_idxs:
            pred = y[idx]
            m = mask[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                wandb.Image(pred),
                wandb.Image(m)
            )