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

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import utils

from auramask.utils.constants import (
    EnumAction,
    InstaFilterEnum,
    FaceEmbedEnum,
    DatasetEnum,
)

# Global hparams object
hparams: dict = {}
keras.config.disable_traceback_filtering()
# Normalize network to use channels last ordering
keras.backend.set_image_data_format("channels_last")

import auramask

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
        prog="AuraMask Testing",
        description="An evaluation script for pairwise comparison",
    )
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--alias", type=str, default="latest")
    parser.add_argument(
        "-F", type=FaceEmbedEnum, nargs="+", required=False, action=EnumAction
    )
    parser.add_argument(
        "--threshold", default=True, type=bool, action=argparse.BooleanOptionalAction
    )
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
            "alex",
            "vgg",
            "squeeze",
            "mse",
            "mae",
            "dsssim",
            "gsssim",
            "nima",
            "ffl",
            "exposure",
            "color",
            "illumination",
            "spatial",
            "style",
            "content",
            "variation",
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
        "--pairs-dataset",
        default="logasja/lfw",
        type=str,
    )
    parser.add_argument(
        "--instagram-filter", type=InstaFilterEnum, action=EnumAction, required=False
    )

    args = parser.parse_args()

    return args


def set_seed():
    seed = hparams["seed"]
    seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % 10**8
    utils.set_random_seed(seed)


def load_model(config: dict, weights_path: str) -> keras.Model:
    model = auramask.AuraMask(config, weights_path)
    return model


def load_pairs_ds() -> datasets.Dataset:
    ds = datasets.load_dataset(hparams["pairs_dataset"], "pairs", split="test")
    insta: InstaFilterEnum = hparams["instagram_filter"]
    dims: tuple[int] = hparams["input"]

    batch = hparams["batch"]

    if insta:
        ds = ds.map(
            lambda x: {"img_0": insta.filter_transform(x)["target"]},
            input_columns=["img_0"],
            batched=True,
            batch_size=batch,
            load_from_cache_file=False,
            num_proc=os.cpu_count(),
        )

    def v_transform(examples):
        examples = DatasetEnum.data_collater(examples, {"w": dims[0], "h": dims[1]})

        pairs = np.stack(examples["pair"], dtype="int8")
        img_0 = np.stack(examples["img_0"], dtype="float32")
        img_1 = np.stack(examples["img_1"], dtype="float32")

        return (pairs, img_0, img_1)

    ds.set_format("numpy")

    from torch.utils.data import DataLoader

    ds = DataLoader(
        ds,
        batch,
        shuffle=False,
        # persistent_workers=True,
        collate_fn=v_transform,
        num_workers=int(os.getenv("DL_TEST_WORKERS", 4)),
        pin_memory=True,
    )

    return ds


def main():
    # Constant Defaults
    hparams["input"] = (256, 256)
    hparams.update(parse_args().__dict__)
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

    # ds = load_pairs_ds()

    # Get the logged weights
    api = wandb.Api(overrides={"entity": "spuds", "project": "auramask"})
    train_run: wandb.Run = api.run(hparams["run_id"])
    model_artifact = None
    for logged_artifact in train_run.logged_artifacts():
        if logged_artifact.type == "model":
            if hparams["alias"] in logged_artifact.aliases + [logged_artifact.version]:
                model_artifact = logged_artifact
                break
    if not model_artifact:
        raise Exception(
            "Unable to find an artifact with alias or version {0}".format(
                hparams["alias"]
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

    run = wandb.init(project="auramask", job_type="evaluation", dir=logdir)
    run.use_model(model_artifact)

    model = load_model(train_run.config, logged_weights)

    print(model.summary())


if __name__ == "__main__":
    main()
