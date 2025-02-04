# ruff: noqa: E402
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import wandb
import auramask
import argparse
from ast import literal_eval
from pathlib import Path
from random import choice
from string import ascii_uppercase
from datetime import datetime
from hashlib import sha256

# Global hparams object
hparams: dict = {}
keras.config.disable_traceback_filtering()
# Normalize network to use channels last ordering
keras.backend.set_image_data_format("channels_last")


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
        prog="AuraMask Training",
        description="A training script for the AuraMask network",
    )
    parser.add_argument(
        "-m",
        "--model-backbone",
        type=auramask.constants.BaseModels,
        action=auramask.constants.EnumAction,
        required=True,
    )
    parser.add_argument("--model-config", type=argparse.FileType("r"))
    parser.add_argument(
        "-F",
        type=auramask.constants.FaceEmbedEnum,
        nargs="+",
        required=False,
        action=auramask.constants.EnumAction,
    )
    parser.add_argument(
        "--threshold", default=True, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("-p", "--rho", type=float, default=1.0)
    parser.add_argument("-a", "--alpha", type=float, default=2e-4)
    parser.add_argument("-e", "--epsilon", type=float, default=0.03)
    parser.add_argument("-B", "--batch-size", dest="batch", type=int, default=32)
    parser.add_argument("-E", "--epochs", type=int, default=5)
    parser.add_argument("-s", "--steps-per-epoch", type=int, default=-1)
    parser.add_argument("-d", "--dims", type=int, default=256)
    parser.add_argument(
        "--lpips-spatial",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--mixed-precision",
        default=True,
        type=bool,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-L",
        "--losses",
        type=str,
        default=["none"],
        choices=[
            "lpips",
            "alex",
            "vgg",
            "squeeze",
            "mse",
            "mae",
            "ssimc",
            "cwssim",
            "dsssim",
            "gsssim",
            "nima",
            "iqanima",
            "psnr",
            "ffl",
            "exposure",
            "color",
            "illumination",
            "spatial",
            "style",
            "content",
            "variation",
            "histogram",
            "topiq",
            "topiqnr",
            "ms_swd",
            "none",
        ],
        nargs="+",
    )
    parser.add_argument("-l", "--lambda", type=float, default=[1.0], nargs="+")
    parser.add_argument(
        "--adaptive-loss",
        type=str,
        required=False,
        choices=["loss-weighted", "normalized", "base"],
    )
    parser.add_argument(
        "--adaptive-loss-frequency", type=str, required=False, default="epoch"
    )
    parser.add_argument(
        "--gradient-alteration", type=str, required=False, choices=["pc-grad"]
    )
    parser.add_argument(
        "--style-ref",
        type=auramask.losses.StyleRefs,
        action=auramask.constants.EnumAction,
        default=auramask.losses.StyleRefs.STARRYNIGHT,
        required=False,
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
    parser.add_argument("--training", type=float, required=True)
    parser.add_argument("--testing", type=float, required=True)
    parser.add_argument(
        "--eager", default=False, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("-v", "--verbose", default=1, type=int)
    parser.add_argument(
        "--note", default=False, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-C",
        "--color-space",
        type=auramask.constants.ColorSpaceEnum,
        action=auramask.constants.EnumAction,
        default=auramask.constants.ColorSpaceEnum.RGB,
        required=False,
    )
    parser.add_argument(
        "--checkpoint", default=False, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-D",
        "--dataset",
        default="lfw",
        type=auramask.constants.DatasetEnum,
        action=auramask.constants.EnumAction,
        required=True,
    )
    parser.add_argument(
        "--instagram-filter",
        type=auramask.constants.InstaFilterEnum,
        action=auramask.constants.EnumAction,
        required=False,
    )

    args = parser.parse_args()

    from json import load

    args.model_config = load(args.model_config)

    return args


def load_data():
    ds: auramask.constants.DatasetEnum = hparams["dataset"]
    train_size, test_size = hparams["training"], hparams["testing"]

    # In the case that a number of samples is passed in instead of a percentage of the test split
    if train_size > 1.0:
        train_size = int(train_size)
    if test_size > 1.0:
        test_size = int(test_size)

    insta: auramask.constants.InstaFilterEnum = hparams["instagram_filter"]
    t_ds, v_ds = ds.load_dataset(
        hparams["input"],
        train_size,
        test_size,
        hparams["batch"],
        insta.filter_transform if insta else None,
    )

    hparams["dataset"] = ds.name.lower()

    return t_ds, v_ds


def initialize_loss():
    losses = []
    weights = []
    loss_config = {}
    cs_transforms = []

    is_not_rgb = hparams["color_space"].name.casefold() != "rgb"
    F = hparams.pop("F")
    threshold = hparams.pop("threshold")
    rho = hparams.pop("rho")
    if F:
        for f in F:
            if threshold:  # Loss with thresholding
                losses.append(
                    auramask.losses.FaceEmbeddingThresholdLoss(
                        f=f,
                        threshold=f.get_threshold(),
                        negative_slope=1.0,
                    )
                )
                weights.append(rho)
            else:  # Loss as described by ReFace
                losses.append(auramask.losses.FaceEmbeddingAbsoluteLoss(f=f))
                weights.append(rho / len(F))
            loss_config[losses[-1].name] = losses[-1].get_config() | {
                "weight": weights[-1]
            }
            cs_transforms.append(
                is_not_rgb
            )  # Determine if it needs to be transformed to rgb space

    if "none" not in hparams["losses"]:
        lam = hparams.pop("lambda")
        loss_in = hparams.pop("losses")
        if len(loss_in) != len(lam) and len(lam) > 1:
            raise argparse.ArgumentError(
                message="The length of lambda values must equal that of losses argument"
            )
        elif len(lam) <= 1:
            w = lam[0] if len(lam) > 0 else 1.0
            iters = zip(loss_in, [w] * len(loss_in))
        else:
            iters = zip(loss_in, lam)

        for loss_i, w_i in iters:
            if loss_i == "mse":
                tmp_loss = keras.losses.MeanSquaredError()
                cs_transforms.append(False)
            elif loss_i == "mae":
                tmp_loss = keras.losses.MeanAbsoluteError()
                cs_transforms.append(False)
            elif loss_i == "ssimc":
                tmp_loss = auramask.losses.IQASSIMC()
                cs_transforms.append(False)
            elif loss_i == "cwssim":
                tmp_loss = auramask.losses.IQACWSSIM()
                cs_transforms.append(False)
            elif loss_i == "dsssim":
                tmp_loss = auramask.losses.DSSIMObjective()
                cs_transforms.append(False)
            elif loss_i == "gsssim":
                tmp_loss = auramask.losses.GRAYSSIMObjective()
                cs_transforms.append(False)
            elif loss_i == "nima":
                tmp_loss = auramask.losses.AestheticLoss(
                    name="NIMA-A", backbone="inceptionresnetv2"
                )
                cs_transforms.append(is_not_rgb)
            elif loss_i == "iqanima":
                tmp_loss = auramask.losses.IQAAestheticLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "exposure":
                tmp_loss = auramask.losses.ExposureControlLoss(mean_val=0.6)
                cs_transforms.append(is_not_rgb)
            elif loss_i == "color":
                tmp_loss = auramask.losses.ColorConstancyLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "illumination":
                tmp_loss = auramask.losses.IlluminationSmoothnessLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "spatial":
                tmp_loss = auramask.losses.SpatialConsistencyLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "style":
                style = hparams.pop("style_ref")
                tmp_loss = auramask.losses.StyleLoss(reference=style)
                cs_transforms.append(is_not_rgb)
            elif loss_i == "variation":
                tmp_loss = auramask.losses.VariationLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "content":
                tmp_loss = auramask.losses.ContentLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "topiq":
                tmp_loss = auramask.losses.SoftTopIQFR(tolerance=0.4)
                cs_transforms.append(is_not_rgb)
            elif loss_i == "topiqnr":
                tmp_loss = auramask.losses.TopIQNR()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "histogram":
                tmp_loss = auramask.losses.HistogramMatchingLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "psnr":
                tmp_loss = auramask.losses.IQAPSNR()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "lpips":
                tmp_loss = auramask.losses.IQAPerceptual()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "ms_swd":
                style = hparams.pop("style_ref")
                tmp_loss = auramask.losses.MSSWD(reference=style)
                cs_transforms.append(is_not_rgb)
            else:
                spatial = hparams.pop("lpips_spatial")
                tmp_loss = auramask.losses.PerceptualLoss(
                    backbone=loss_i, spatial=spatial, tolerance=0.05
                )
                cs_transforms.append(is_not_rgb)

            losses.append(tmp_loss)
            weights.append(w_i)
            loss_config[tmp_loss.name] = tmp_loss.get_config() | {"weight": w_i}

    if not is_not_rgb:
        cs_transforms = None

    hparams["losses"] = loss_config

    return losses, weights, cs_transforms


def initialize_model():
    losses, losses_w, losses_t = initialize_loss()

    adaptive_callback = []

    # Allows modifying the config at calling with the AURAMASK_CONFIG environment variable
    cfg_mod: dict = literal_eval(os.getenv("AURAMASK_CONFIG", "{}"))
    for key, val in cfg_mod.items():
        if isinstance(val, str) and val.lower() in ["true", "false"]:
            cfg_mod[key] = True if val.lower() == "true" else False
    hparams["model_config"].update(cfg_mod)

    hparams["model"] = hparams.pop("model_backbone").name.lower()
    model = auramask.AuraMask(hparams)

    # keras.utils.plot_model(model, expand_nested=True, show_shapes=True)

    optimizer = keras.optimizers.Adam(learning_rate=hparams["alpha"], clipnorm=1.0)

    if hparams["adaptive_loss"] is not None:
        try:
            freq = int(hparams["adaptive_loss_frequency"])
        except ValueError:
            freq = hparams["adaptive_loss_frequency"]

        adaptive_callback = [
            auramask.callbacks.AdaptiveLossCallback(
                [lss.name for lss in losses],
                weights=losses_w,
                frequency=freq,
                algorithm=hparams["adaptive_loss"],
                clip_weights=True,
                backup_dir=os.path.join(hparams["log_dir"], "backup"),
            )
        ]

    if hparams["gradient_alteration"] is not None:
        grad_fn = auramask.pcgrad.compute_pc_grads
    else:
        grad_fn = None

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=losses_w,
        run_eagerly=hparams.pop("eager"),
        jit_compile=False,
        auto_scale_loss=True,
        gradient_alter=grad_fn,
    )

    return model, adaptive_callback


def set_seed():
    seed = hparams["seed"]
    seed = int(sha256(seed.encode("utf-8")).hexdigest(), 16) % 10**8
    keras.utils.set_random_seed(seed)


def get_sample_data(ds):
    if keras.backend.backend() == "tensorflow":
        for x in ds.take(1):
            inp = x[0]
    else:
        for batch in ds:
            inp = batch[0]
            break

    return inp


def init_callbacks(hparams: dict, sample, logdir, note: str = ""):
    checkpoint = hparams.pop("checkpoint")
    tmp_hparams = hparams
    tmp_hparams["color_space"] = (
        tmp_hparams["color_space"].name if tmp_hparams["color_space"] else "rgb"
    )
    tmp_hparams["input"] = str(tmp_hparams["input"])
    tmp_hparams["task_id"] = str(os.getenv("SLURM_JOB_ID", None))

    train_callbacks = []
    wandb.init(
        project="auramask",
        id=os.getenv("WANDB_RUN_ID", None),
        dir=logdir,
        config=tmp_hparams,
        name=os.getenv("SLURM_JOB_NAME", None),
        notes=note,
        resume="allow",
    )

    train_callbacks.append(
        keras.callbacks.BackupAndRestore(backup_dir=os.path.join(logdir, "backup"))
    )

    if checkpoint:
        train_callbacks.append(
            auramask.callbacks.AuramaskCheckpoint(
                filepath=os.path.join(logdir, "checkpoints"),
                freq_mode="epoch",
                save_weights_only=True,
                save_freq=int(os.getenv("AURAMASK_CHECKPOINT_FREQ", 100)),
            )
        )

    train_callbacks.append(auramask.callbacks.AuramaskWandbMetrics(log_freq="epoch"))
    train_callbacks.append(
        auramask.callbacks.AuramaskCallback(
            validation_data=sample,
            data_table_columns=["idx", "orig", "aug"],
            pred_table_columns=["epoch", "idx", "pred", "mask"],
            log_freq=int(os.getenv("AURAMASK_LOG_FREQ", 5)),
        )
    )
    # train_callbacks.append(
    #     keras.callbacks.ReduceLROnPlateau(
    #         monitor="val_loss", patience=10, verbose=1, cooldown=50, min_lr=2e-9,
    #     )
    # )
    train_callbacks.append(auramask.callbacks.AuramaskStopOnNaN())
    return train_callbacks


def main():
    # Constant Defaults
    hparams["optimizer"] = "adam"
    hparams.update(parse_args().__dict__)
    dims = hparams.pop("dims")
    hparams["input"] = (dims, dims)
    log = hparams.pop("log")
    logdir = hparams["log_dir"]
    note = hparams.pop("note")
    verbose = hparams.pop("verbose")
    mixed_precision = hparams["mixed_precision"]

    if mixed_precision:
        print("Using mixed precision for training")
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
    hparams["log_dir"] = str(logdir)

    set_seed()
    # Load the training and validation data
    t_ds, v_ds = load_data()

    model, callbacks = initialize_model()

    v = get_sample_data(v_ds)
    model(v)

    callbacks.extend(init_callbacks(hparams, v, hparams.pop("log_dir"), note))

    training_history = model.fit(
        t_ds,
        callbacks=callbacks,
        epochs=hparams["epochs"],
        verbose=verbose,
        validation_data=v_ds,
        steps_per_epoch=hparams["steps_per_epoch"],
    )
    return training_history


if __name__ == "__main__":
    main()
