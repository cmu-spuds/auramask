import io
from os import PathLike, getenv, path
import os
import string
from typing import Any, Dict, List, Literal, Optional

import wandb
from wandb.sdk.lib import telemetry

from keras import callbacks as k_callbacks
from wandb.integration.keras import (
    WandbMetricsLogger,
    WandbEvalCallback,
)
from wandb.integration.keras.callbacks.model_checkpoint import SaveStrategy
from keras import preprocessing, ops


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
        self.x = validation_data[:num_samples]
        self.log_freq = log_freq
        self.__cur_epoch = 0

    def on_train_begin(self, logs: Dict[SaveStrategy, float] | None = None) -> None:
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
        # return super().on_train_begin(logs)

    def on_epoch_end(
        self, epoch: int, logs: Dict[SaveStrategy, float] | None = None
    ) -> None:
        if epoch % self.log_freq == 0:
            self.save_results()
        self.__cur_epoch = epoch

    def save_results(self):
        y, mask = self.model(self.x, training=False)
        if mask.shape[-1] > 3 and mask.shape[-1] % 3 == 0:
            data = {}
            r = ops.split(mask, 8, axis=-1)
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
        # for idx, (orig, embeds, names) in enumerate(zip(self.x, self.y)):
        #     self.data_table.add_data(idx, wandb.Image(orig), wandb.Table(data=embeds[idx]))         #TODO: save all pre-computed embeddings to table y: ((N_EMBEDS), (N_NAMES))

    def add_model_predictions(
        self, epoch: int, logs: Dict[str, float] | None = None
    ) -> None:
        pass

    def on_train_end(self, logs: Dict[str, float] | None = None) -> None:
        self.save_results()


class AuramaskCheckpoint(k_callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath: str | PathLike[str],
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: Literal["auto"] | Literal["min"] | Literal["max"] = "auto",
        save_freq: int = 1,
        freq_mode: Literal["batch"] | Literal["epoch"] = "epoch",
        initial_value_threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        # self._bk_filepath = filepath
        if save_weights_only:
            filepath = os.path.join(filepath, "{epoch:02d}-{val_loss:.2f}.weights.h5")
        else:
            filepath = os.path.join(filepath, "{epoch:02d}-{val_loss:.2f}.keras")

        if freq_mode == "epoch":
            super().__init__(
                filepath,
                monitor,
                verbose,
                save_best_only,
                save_weights_only,
                mode,
                "epoch",
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
                initial_value_threshold,
                **kwargs,
            )

        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before `WandbModelCheckpoint()`"
            )
        with telemetry.context(run=wandb.run) as tel:
            tel.feature.keras_model_checkpoint = True

        self.save_weights_only = save_weights_only

        # User-friendly warning when trying to save the best model.
        if self.save_best_only:
            self._check_filepath()

        self.__freq = save_freq
        self.__cur_epoch = 0

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        # if self.model.name == "AuraMask":
        #     self.train_wrapper = self.model
        #     self.set_model(self.model.model)

    def on_epoch_end(
        self, epoch: int, logs: Dict[SaveStrategy, float] | None = None
    ) -> None:
        if self.save_freq == "epoch":
            # self.train_wrapper.save(os.path.join(self._bk_filepath, "training_state.keras"), overwrite=True)
            if epoch > 0 and epoch % self.__freq == 0:
                self._on_epoch_end(epoch, logs)
        self.__cur_epoch = epoch

    def on_train_end(self, logs: Dict[SaveStrategy, float] | None = None) -> None:
        self.on_epoch_end(self.__cur_epoch, logs)

    def on_train_batch_end(
        self, batch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        if self._should_save_on_batch(batch):
            # Save the model and get filepath
            self._save_model(epoch=self._current_epoch, batch=batch, logs=logs)
            filepath = self._get_file_path(
                epoch=self._current_epoch, batch=batch, logs=logs
            )
            # Log the model as artifact
            aliases = ["latest", f"epoch_{self._current_epoch}_batch_{batch}"]
            self._log_ckpt_as_artifact(filepath, aliases=aliases)

    def _on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        super().on_epoch_end(epoch, logs)
        # Check if model checkpoint is created at the end of epoch.
        if self.save_freq == "epoch":
            # Get filepath where the model checkpoint is saved.
            filepath = self._get_file_path(epoch=epoch, batch=None, logs=logs)
            # Log the model as artifact
            aliases = ["latest", f"epoch_{epoch}"]
            self._log_ckpt_as_artifact(filepath, aliases=aliases)

    def _log_ckpt_as_artifact(
        self, filepath: str, aliases: Optional[List[str]] = None
    ) -> None:
        """Log model checkpoint as  W&B Artifact."""
        try:
            assert wandb.run is not None
            model_checkpoint_artifact = wandb.Artifact(
                f"run_{wandb.run.id}_model", type="model"
            )
            if os.path.isfile(filepath):
                model_checkpoint_artifact.add_file(filepath)
            elif os.path.isdir(filepath):
                model_checkpoint_artifact.add_dir(filepath)
            else:
                raise FileNotFoundError(f"No such file or directory {filepath}")
            wandb.log_artifact(model_checkpoint_artifact, aliases=aliases or [])
        except ValueError:
            # This error occurs when `save_best_only=True` and the model
            # checkpoint is not saved for that epoch/batch. Since TF/Keras
            # is giving friendly log, we can avoid clustering the stdout.
            pass

    def _check_filepath(self) -> None:
        placeholders = []
        for tup in string.Formatter().parse(self.filepath):
            if tup[1] is not None:
                placeholders.append(tup[1])
        if len(placeholders) == 0:
            wandb.termwarn(
                "When using `save_best_only`, ensure that the `filepath` argument "
                "contains formatting placeholders like `{epoch:02d}` or `{batch:02d}`. "
                "This ensures correct interpretation of the logged artifacts.",
                repeat=False,
            )


def init_callbacks(hparams: dict, sample, logdir, note: str = ""):
    checkpoint = hparams.pop("checkpoint")
    tmp_hparams = hparams
    tmp_hparams["color_space"] = (
        tmp_hparams["color_space"].name if tmp_hparams["color_space"] else "rgb"
    )
    tmp_hparams["input"] = str(tmp_hparams["input"])
    tmp_hparams["task_id"] = str(getenv("SLURM_JOB_ID", None))

    callbacks = []
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
                save_weights_only=True,
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
