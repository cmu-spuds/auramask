import io
from os import PathLike, getenv, path
from typing import Any, Dict, Literal

import wandb
from keras import callbacks as k_callbacks
from wandb.integration.keras import (
    WandbMetricsLogger,
    WandbEvalCallback,
    WandbModelCheckpoint,
)
from wandb.integration.keras.callbacks.model_checkpoint import SaveStrategy
from keras import preprocessing


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
        self.x = validation_data[0]
        self.y = validation_data[1]
        self.log_freq = log_freq
        self.__cur_epoch = 0

    def on_train_begin(self, logs: Dict[SaveStrategy, float] | None = None) -> None:
        wandb.log(
            {
                "image": [
                    wandb.Image(preprocessing.image.array_to_img(x_i))
                    for x_i in self.x[:5]
                ],
            },
            step=0,
        )
        return super().on_train_begin(logs)

    def on_epoch_end(
        self, epoch: int, logs: Dict[SaveStrategy, float] | None = None
    ) -> None:
        if epoch % self.log_freq == 0:
            super().on_epoch_end(epoch, logs)
        self.__cur_epoch = epoch

    def add_ground_truth(self, logs: Dict[str, float] | None = None) -> None:
        pass
        # for idx, (orig, embeds, names) in enumerate(zip(self.x, self.y)):
        #     self.data_table.add_data(idx, wandb.Image(orig), wandb.Table(data=embeds[idx]))         #TODO: save all pre-computed embeddings to table y: ((N_EMBEDS), (N_NAMES))

    def add_model_predictions(
        self, epoch: int, logs: Dict[str, float] | None = None
    ) -> None:
        y, mask = self.model(self.x, training=False)
        table_idxs = self.data_table_ref.get_index()

        if mask.shape[-1] > 3 and mask.shape[-1] % 3 == 0:
            data = {}
            for i in range(0, mask.shape[-1], 3):
                data["r%d" % (i / 3)] = [
                    wandb.Image(preprocessing.image.array_to_img(m_i))
                    for m_i in mask[:, :, :, i : i + 3]
                ]

            data["image"] = [
                wandb.Image(preprocessing.image.array_to_img(y_i)) for y_i in y[:]
            ]
            wandb.log(data, step=wandb.run.step)

            # for idx in table_idxs:
            #     pred = y[idx]
            #     m = mask[idx]
            #     self.pred_table.add_data(
            #         epoch,
            #         self.data_table_ref.data[idx][0],
            #         wandb.Image(pred),
            #         wandb.Image(m),
            #     )
        else:
            wandb.log(
                {
                    "image": [
                        wandb.Image(preprocessing.image.array_to_img(y_i))
                        for y_i in y[:]
                    ],
                    "mask": [
                        wandb.Image(preprocessing.image.array_to_img(m_i))
                        for m_i in mask[:]
                    ],
                },
                step=wandb.run.step,
            )

            for idx in table_idxs:
                pred = y[idx]
                m = mask[idx]
                self.pred_table.add_data(
                    epoch,
                    self.data_table_ref.data[idx][0],
                    wandb.Image(pred),
                    wandb.Image(m),
                )

    def on_train_end(self, logs: Dict[str, float] | None = None) -> None:
        super().on_epoch_end(self.__cur_epoch, logs=logs)


class AuramaskCheckpoint(WandbModelCheckpoint):
    def __init__(
        self,
        filepath: str | PathLike[str],
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: Literal["auto"] | Literal["min"] | Literal["max"] = "auto",
        save_freq: int = 1,
        options: str | None = None,
        initial_value_threshold: float | None = None,
        freq_mode: Literal["batch"] | Literal["epoch"] = "epoch",
        **kwargs: Any,
    ) -> None:
        if freq_mode == "epoch":
            super().__init__(
                filepath,
                monitor,
                verbose,
                save_best_only,
                save_weights_only,
                mode,
                "epoch",
                options,
                initial_value_threshold,
                **kwargs,
            )
        else:
            super().__init__(
                filepath,
                monitor,
                verbose,
                save_best_only,
                save_weights_only,
                mode,
                save_freq,
                options,
                initial_value_threshold,
                **kwargs,
            )

        self.__freq = save_freq
        self.__cur_epoch = 0

    def on_epoch_end(
        self, epoch: int, logs: Dict[SaveStrategy, float] | None = None
    ) -> None:
        if self.save_freq == "epoch":
            if epoch > 0 and epoch % self.__freq == 0:
                super().on_epoch_end(epoch, logs)
        self.__cur_epoch = epoch

    def on_train_end(self, logs: Dict[SaveStrategy, float] | None = None) -> None:
        super().on_epoch_end(self.__cur_epoch, logs)


def init_callbacks(hparams: dict, sample, logdir, note: str = ""):
    checkpoint = hparams.pop("checkpoint")
    tmp_hparams = hparams
    tmp_hparams["color_space"] = (
        tmp_hparams["color_space"].name if tmp_hparams["color_space"] else "rgb"
    )
    tmp_hparams["input"] = str(tmp_hparams["input"])

    callbacks = []
    if getenv("WANDB_MODE") != "offline":
        wandb.init(
            project="auramask",
            id=getenv("WANDB_RUN_ID", None),
            dir=logdir,
            config=tmp_hparams,
            name=getenv("SLURM_JOB_NAME", None),
            notes=note,
            resume="allow",
        )

        callbacks.append(
            k_callbacks.BackupAndRestore(backup_dir=path.join(logdir, "backup"))
        )

        if checkpoint:
            callbacks.append(
                AuramaskCheckpoint(
                    filepath=path.join(logdir, "checkpoints"),
                    freq_mode="epoch",
                    save_weights_only=False,
                    save_freq=int(getenv("AURAMASK_CHECKPOINT_FREQ", 100)),
                )
            )
        callbacks.append(WandbMetricsLogger(log_freq="epoch"))
        callbacks.append(
            AuramaskCallback(
                validation_data=sample,
                data_table_columns=["idx", "orig", "aug"],
                pred_table_columns=["epoch", "idx", "pred", "mask"],
                log_freq=int(getenv("AURAMASK_LOG_FREQ", 5)),
            )
        )
    # callbacks.append(LearningRateScheduler())
    return callbacks
