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
from keras import ops
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
        "--with-image",
        default=False,
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
        "--crop-faces", type=str, default="", choices=["", "before", "after"]
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
        default="logasja/lfw",
        type=str,
    )
    parser.add_argument(
        "-V",
        "--validation",
        default="validation",
        type=str,
        nargs=2,
        help="The config and split for the validation dataaset.",
    )
    parser.add_argument(
        "-P",
        "--pairs",
        required=False,
        type=str,
        nargs=3,
        help="The columns mapping to pair groundtruth, first pair image, second pair image.",
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
        model = keras.saving.load_model(f"hf://{hparams['hf_model']}")
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
                    return ops.mean(ops.square(y_true - y_pred), axis=[1, 2, 3])

                tmp_metric = keras.metrics.MeanMetricWrapper(
                    mse,
                    name=metric,
                )
            elif metric == "mae":

                def mae(y_true, y_pred):
                    return ops.mean(ops.absolute(y_true - y_pred), axis=[1, 2, 3])

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
                    metrics.append(aura_metrics.CosineValidation(f=f))
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
    v_config, v_split = hparams["validation"]
    ds = datasets.load_dataset(
        hparams["dataset"], name=v_config, split=v_split, num_proc=8
    )
    pair, img_a, img_b = hparams["pairs"]
    ds = ds.rename_columns({pair: "pair", img_a: "A", img_b: "B"}).select_columns(
        ["pair", "A", "B"]
    )

    dims: tuple[int] = hparams["input"]

    process_image = A.Compose(
        [
            A.ToFloat(max_value=255, p=1),
            # A.CenterCrop(dims[0], dims[1]),
            A.LongestMaxSize(
                np.maximum(dims[0], dims[1]), interpolation=A.cv2.INTER_AREA
            ),
        ]
    )

    def transform(batch):
        A = np.stack(
            [
                process_image(image=keras.utils.img_to_array(img, dtype="uint8"))[
                    "image"
                ]
                for img in batch["A"]
            ]
        )
        B = np.stack(
            [
                process_image(image=keras.utils.img_to_array(img, dtype="uint8"))[
                    "image"
                ]
                for img in batch["B"]
            ]
        )
        return {
            "pair": batch["pair"],
            "A": A,
            "B": B,
        }

    ds = ds.with_transform(transform)
    return ds


def batch_crop_to_face(img_batch, detector, max_value):
    if max_value == 1:
        img_batch = ops.cast(img_batch * 255, "uint8")  # Scale to [0,255] pixel space

    if keras.backend.backend() == "tensorflow":
        batch_faces = []
        for img in img_batch:
            face = detector.detect_faces(
                img,
                box_format="xywh",
                min_face_size=15,  # Detect smaller faces
                threshold_pnet=0.5,  # More proposals from PNet
                threshold_rnet=0.6,  # Loosen RNet filtering
                threshold_onet=0.7,  # More final faces accepted by ONet
            )
            if len(face) > 0:
                bbox = face[0]["box"]
                img = ops.image.crop_images(
                    img,
                    left_cropping=bbox[0],
                    top_cropping=bbox[1],
                    target_width=bbox[2],
                    target_height=bbox[3],
                )
            img = ops.image.resize(img, (224, 224))
            batch_faces.append(img / 255.0)
        return ops.stack(batch_faces)
    elif keras.backend.backend() == "torch":
        out = ops.stack(detector(ops.convert_to_numpy(img_batch)))
        img_batch = ops.transpose(out, (0, 2, 3, 1))  # convert to channels last
        return img_batch


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

    wandb.init(
        project="auramask",
        id=os.getenv("WANDB_RUN_ID", None),
        dir=logdir,
        name=os.getenv("WANDB_RUN_NAME", None),
        notes=note,
        resume="allow",
        job_type="evaluation",
        group=os.getenv("WANDB_RUN_GROUP", None),
    )

    ds = load_dataset()

    metrics = initialize_metrics()

    if hparams["hf_model"] or hparams["run_id"]:
        model = load_model()
    else:
        model = None

    if hparams["with_image"]:
        validation_tab = wandb.Table(
            ["pair", "face a", "face b"] + [m.name for m in metrics]
        )
    else:
        validation_tab = wandb.Table(["pair"] + [m.name for m in metrics])

    wandb.run.config.update(hparams)

    if hparams["crop_faces"] != "":
        if keras.backend.backend() == "tensorflow":
            from mtcnn.mtcnn import MTCNN

            detector = MTCNN()
        elif keras.backend.backend() == "torch":
            from facenet_pytorch import MTCNN

            detector = MTCNN(image_size=dims, margin=14)

    for example in (
        _ := tqdm.tqdm(
            ds.iter(batch_size=hparams["batch"]),
            total=int(np.ceil(ds.num_rows / hparams["batch"])),
            position=0,
        )
    ):
        batch_a = ops.stop_gradient(ops.convert_to_tensor(example["A"]))
        batch_b = ops.convert_to_tensor(example["B"])

        if hparams["crop_faces"] == "before":
            batch_a = batch_crop_to_face(batch_a, detector, max_value=1)
            batch_b = batch_crop_to_face(batch_b, detector, max_value=1)

        if model:
            batch_a = ops.stop_gradient(model(batch_a, training=False)[0])

        if hparams["crop_faces"] == "after":
            batch_a = batch_crop_to_face(batch_a, detector, max_value=1)
            batch_b = batch_crop_to_face(batch_b, detector, max_value=1)

        met_vals = []
        for metric in metrics:
            values = metric._fn(
                batch_a, batch_b, **metric._fn_kwargs
            )  # Hacky way to compute a non-mean evaluation for saving
            keras.metrics.Mean.update_state(
                metric, values
            )  # Hacky way to update state without having to recompute
            met_vals.append([v for v in ops.convert_to_numpy(ops.squeeze(values))])
            metric.reset_state()

        B = ops.shape(batch_a)[0]

        if hparams["with_image"]:
            for i in range(B):
                # print([v[i] for v in met_vals])
                validation_tab.add_data(
                    example["pair"][i],
                    wandb.Image(keras.utils.array_to_img(batch_a[i])),
                    wandb.Image(keras.utils.array_to_img(batch_b[i])),
                    *[v[i] for v in met_vals],
                )
        else:
            for i in range(B):
                # print([v[i] for v in met_vals])
                validation_tab.add_data(
                    example["pair"][i],
                    *[v[i] for v in met_vals],
                )

    wandb.run.log({"validation": validation_tab})

    wandb.finish()


if __name__ == "__main__":
    main()
