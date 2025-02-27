# ruff: noqa: E402
import argparse
import hashlib
import os
from pathlib import Path
from random import choice
from string import ascii_uppercase
from datetime import datetime

import wandb
import datasets
import tqdm

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import albumentations as A
from auramask import metrics as aura_metrics

from auramask.utils.constants import (
    EnumAction,
    FaceEmbedEnum,
)

# Global hparams object
hparams: dict = {}
keras.config.disable_traceback_filtering()
# Normalize network to use channels last ordering
keras.backend.set_image_data_format("channels_last")

ARTIFACT_URL = "run_{0}_model:{1}"


# Path checking and creation if appropriate
def dir_path(path):
    if path:
        path = Path(path)
        try:
            if not path.parent.parent.exists():
                raise FileNotFoundError()
            path.mkdir(parents=True, exist_ok=True)
            return str(path.absolute())
        except FileNotFoundError:
            raise argparse.ArgumentTypeError(
                f"The directory {path} cannot have more than 2 missing parents."
            )
        except FileExistsError:
            raise argparse.ArgumentTypeError(f"The directory {path} exists as a file")
    return


def parse_args():
    parser = argparse.ArgumentParser(
        prog="AuraMask Validation",
        description="An evaluation script to compute metrics on validation data.",
    )
    parser.add_argument("--run-id", type=str, required=False)
    parser.add_argument("--hf-model", type=str, required=False)
    parser.add_argument("--version", type=str, default="latest")
    parser.add_argument(
        "-F", type=FaceEmbedEnum, nargs="+", required=False, action=EnumAction
    )
    parser.add_argument("-d", "--dims", type=int, default=256)
    parser.add_argument("-B", "--batch-size", dest="batch", type=int, default=32)
    parser.add_argument(
        "--mixed-precision",
        default=True,
        type=bool,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-M",
        "--metrics",
        type=str,
        default=["none"],
        choices=[
            "mse",
            "mae",
            "lpips",
            "alex",
            "ssimc",
            "dssim",
            "topiq_fr",
            "topiq_nr",
            "cwssim",
            "facevalidation",
            "cosine",
            "euclidean",
            "euclidean_l2",
            "none",
        ],
        nargs="+",
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=str,
        default="".join(choice(ascii_uppercase) for _ in range(12)),
    )
    parser.add_argument(
        "--log", default=True, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--log-dir", default=None, type=dir_path)
    parser.add_argument("-v", "--verbose", default=1, type=int)
    parser.add_argument(
        "--note", default=False, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-D",
        "--dataset",
        default="logasja/fdf",
        type=str,
    )
    parser.add_argument(
        "-V",
        "--validation",
        default="validation",
        type=str,
        nargs=3,
        help="The config, split, and column that maps to validation images.",
    )
    parser.add_argument(
        "-P",
        "--predictions",
        required=False,
        type=str,
        nargs=3,
        help="(Optional) The config, split, and column that maps to precomputed images.",
    )

    args = parser.parse_args()

    return args


def set_seed():
    seed = hparams["seed"]
    seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % 10**8
    keras.utils.set_random_seed(seed)


def load_model() -> keras.Model:
    if hparams["run_id"] and hparams["hf_model"]:
        raise ValueError("Either choose run_id or hf-model")
    elif hparams["run_id"]:
        # Get the logged weights
        api = wandb.Api(overrides={"entity": "spuds", "project": "auramask"})
        train_run: wandb.Run = api.run(hparams["run_id"])
        model_artifact = None
        for logged_artifact in train_run.logged_artifacts():
            if logged_artifact.type == "model":
                if hparams["version"] in logged_artifact.aliases + [
                    logged_artifact.version
                ]:
                    model_artifact = logged_artifact
                    break
        if not model_artifact:
            raise Exception(
                "Unable to find an artifact with alias or version {0}".format(
                    hparams["version"]
                )
            )
        logged_weights = None
        for file in model_artifact.manifest.entries.keys():
            if "h5" in file:
                logged_weights = model_artifact.get_entry(file)
                break
            elif "keras" in file:
                logged_weights = model_artifact.get_entry(file)
                break
        logged_weights = logged_weights.download()
        wandb.use_model(model_artifact)
        model = keras.Model.from_config(train_run.config["model_config"])
        model.load_weights(logged_weights)
    elif hparams["hf_model"]:
        model = keras.saving.load_model(f"hf://{hparams["hf_model"]}")
    else:
        raise ValueError("No value given for run_id or for hf-model")
    return model


def initialize_metrics() -> list[keras.Metric]:
    metrics = []
    metric_config = {}

    F = hparams.pop("F")

    if "none" not in hparams["metrics"]:
        metrics_in = hparams.pop("metrics")

        for metric in metrics_in:
            if metric == "mse":

                def mse(y_true, y_pred):
                    return keras.ops.mean(
                        keras.ops.square(y_true - y_pred), axis=[1, 2, 3]
                    )

                tmp_metric = keras.metrics.MeanMetricWrapper(
                    mse,
                    name=metric,
                )
            elif metric == "mae":

                def mae(y_true, y_pred):
                    return keras.ops.mean(
                        keras.ops.absolute(y_true - y_pred), axis=[1, 2, 3]
                    )

                tmp_metric = keras.metrics.MeanMetricWrapper(
                    mae,
                    name=metric,
                )
            elif metric == "ssimc":
                tmp_metric = aura_metrics.IQASSIMC()
            elif metric == "cwssim":
                tmp_metric = aura_metrics.IQACWSSIM()
            elif metric == "dssim":
                tmp_metric = aura_metrics.DSSIMObjective()
            elif metric == "topiq_fr":
                tmp_metric = aura_metrics.TOPIQFR()
            elif metric == "topiq_nr":
                tmp_metric = aura_metrics.TOPIQNR()
            elif metric == "lpips":
                tmp_metric = aura_metrics.IQAPerceptual()
            elif metric == "alex":
                tmp_metric = aura_metrics.PerceptualSimilarity(backbone="alex")
            elif metric == "facevalidation":
                if not F:
                    raise Exception(
                        "Face validation requires embeddings to be chosen with -F"
                    )
                for f in F:
                    metrics.append(
                        aura_metrics.FaceValidationMetrics(
                            f=f, threshold=f.get_threshold()
                        )
                    )
                    metric_config[metrics[-1].name] = metrics[-1].get_config()
                continue
            elif metric == "cosine":
                if not F:
                    raise Exception(
                        "Embedding distance requires embeddings to be chosen with -F"
                    )
                for f in F:
                    metrics.append(aura_metrics.CosineDistance(f=f))
                    metric_config[metrics[-1].name] = metrics[-1].get_config()
                continue
            elif metric == "euclidean":
                if not F:
                    raise Exception(
                        "Embedding distance requires embeddings to be chosen with -F"
                    )
                for f in F:
                    metrics.append(aura_metrics.EuclideanDistance(f=f))
                    metric_config[metrics[-1].name] = metrics[-1].get_config()
                continue
            elif metric == "euclidean_l2":
                if not F:
                    raise Exception(
                        "Embedding distance requires embeddings to be chosen with -F"
                    )
                for f in F:
                    metrics.append(aura_metrics.EuclideanL2Distance(f=f))
                    metric_config[metrics[-1].name] = metrics[-1].get_config()
                continue
            else:
                raise Exception("Metric not recognized")

            metrics.append(tmp_metric)
            metric_config[tmp_metric.name] = tmp_metric.get_config()

    hparams["metrics"] = metric_config

    return metrics


def load_dataset() -> datasets.Dataset:
    v_config, v_split, v_column = hparams["validation"]
    ds = (
        datasets.load_dataset(
            hparams["dataset"], name=v_config, split=v_split, num_proc=8
        )
        .rename_column(v_column, "target")
        .select_columns(["path", "target"])
    )

    dims: tuple[int] = hparams["input"]

    process_image = A.Compose(
        [
            A.ToFloat(max_value=255, p=1),
            A.LongestMaxSize(np.maximum(dims[0], dims[1])),
        ]
    )

    if hparams["predictions"]:
        p_config, p_split, p_column = hparams["predictions"]
        v_ds = (
            datasets.load_dataset(
                hparams["dataset"], name=p_config, split=p_split, num_proc=8
            )
            .rename_column(p_column, "prediction")
            .select_columns("prediction")
        )
        ds = datasets.concatenate_datasets([ds, v_ds], axis=1).filter(
            lambda x: x["prediction"] is not None
        )

        def transform(batch):
            true_col = np.stack(
                [
                    process_image(image=keras.utils.img_to_array(img, dtype="uint8"))[
                        "image"
                    ]
                    for img in batch["target"]
                ]
            )
            predictions = np.stack(
                [
                    process_image(image=keras.utils.img_to_array(img, dtype="uint8"))[
                        "image"
                    ]
                    for img in batch["prediction"]
                ]
            )
            return {
                "path": batch["path"],
                "target": true_col,
                "prediction": predictions,
            }
    else:  # Load the model for computing filtered outputs if not precomputed in dataset

        def transform(batch):
            true_col = np.stack(
                [
                    process_image(image=keras.utils.img_to_array(img, dtype="uint8"))[
                        "image"
                    ]
                    for img in batch["target"]
                ]
            )
            return {"path": batch["path"], "target": true_col}

    ds = ds.with_transform(transform)
    return ds


def main():
    hparams.update(parse_args().__dict__)
    dims = hparams.pop("dims")
    hparams["input"] = (dims, dims)
    log = hparams.pop("log")
    logdir = hparams.pop("log_dir")
    note = hparams.pop("note")
    mixed_precision = hparams.pop("mixed_precision")

    if mixed_precision:
        print("Using mixed precision")
        keras.mixed_precision.set_dtype_policy("mixed_float16")

    if not log:
        os.environ["WANDB_MODE"] = "offline"

    if note:
        note = input("Note for Run:")
    else:
        note = ""
    if not logdir:
        logdir = Path(
            os.path.join(
                os.path.curdir,
                "logs",
                datetime.now().strftime("%m-%d"),
                hparams["seed"],
            )
        )
    else:
        logdir = Path(os.path.join(logdir))
    logdir.mkdir(parents=True, exist_ok=True)
    logdir = str(logdir)

    set_seed()

    wandb.init(project="auramask", job_type="evaluation", dir=logdir)

    ds = load_dataset()

    metrics = initialize_metrics()

    if hparams["predictions"]:
        model = None
    else:
        model = load_model()

    compute_output = (model is not None) and (hparams["predictions"] is None)

    validation_tab = wandb.Table(["image_name"] + [m.name for m in metrics])

    wandb.run.config.update(hparams)

    for example in (
        pbar := tqdm.tqdm(
            ds.iter(batch_size=hparams["batch"]),
            total=int(np.ceil(ds.num_rows / hparams["batch"])),
            position=0,
        )
    ):
        target_batch = keras.ops.stop_gradient(
            keras.ops.convert_to_tensor(example["target"])
        )
        if compute_output:
            pred_batch = keras.ops.stop_gradient(model(target_batch, training=False)[0])
        else:
            pred_batch = keras.ops.stop_gradient(
                keras.ops.convert_to_tensor(example["prediction"])
            )

        met_vals = []
        for metric in metrics:
            values = metric._fn(
                target_batch, pred_batch, **metric._fn_kwargs
            )  # Hacky way to compute a non-mean evaluation for saving
            keras.metrics.Mean.update_state(
                metric, values
            )  # Hacky way to update state without having to recompute
            met_vals.append(
                [v for v in keras.ops.convert_to_numpy(keras.ops.squeeze(values))]
            )
            metric.reset_state()

        B = keras.ops.shape(target_batch)[0]

        for i in range(B):
            validation_tab.add_data(
                example["path"][i],
                *[v[i] for v in met_vals],
            )

    wandb.run.log({"validation": validation_tab})

    wandb.finish()


if __name__ == "__main__":
    main()
