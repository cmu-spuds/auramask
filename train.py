# ruff: noqa: E402
import argparse
import hashlib
import os
import ast
from pathlib import Path
from random import choice
from string import ascii_uppercase
from datetime import datetime

os.environ["KERAS_BACKEND"] = "torch"

import keras

from auramask.utils.constants import (
    EnumAction,
    DatasetEnum,
    BaseModels,
    ColorSpaceEnum,
    InstaFilterEnum,
    FaceEmbedEnum,
)
from auramask.callbacks import init_callbacks

from auramask.losses import (
    ContentLoss,
    PerceptualLoss,
    FaceEmbeddingLoss,
    FaceEmbeddingThresholdLoss,
    AestheticLoss,
    DSSIMObjective,
    GRAYSSIMObjective,
    StyleLoss,
    StyleRefs,
    VariationLoss,
    ColorConstancyLoss,
    SpatialConsistencyLoss,
    ExposureControlLoss,
    IlluminationSmoothnessLoss,
)

from keras import optimizers as opts, losses as ls, activations, ops, utils

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
        type=BaseModels,
        action=EnumAction,
        required=True,
    )
    parser.add_argument("--model-config", type=argparse.FileType("r"))
    parser.add_argument(
        "-F", type=FaceEmbedEnum, nargs="+", required=False, action=EnumAction
    )
    parser.add_argument(
        "--threshold", default=True, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("-p", "--rho", type=float, default=1.0)
    parser.add_argument("-a", "--alpha", type=float, default=2e-4)
    parser.add_argument("-e", "--epsilon", type=float, default=0.03)
    parser.add_argument("-l", "--lambda", type=float, default=[1.0], nargs="+")
    parser.add_argument("-B", "--batch-size", dest="batch", type=int, default=32)
    parser.add_argument("-E", "--epochs", type=int, default=5)
    parser.add_argument("-s", "--steps-per-epoch", type=int, default=-1)
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
        "--style-ref",
        type=StyleRefs,
        action=EnumAction,
        default=StyleRefs.STARRYNIGHT,
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
        "-C", "--color-space", type=ColorSpaceEnum, action=EnumAction, required=True
    )
    parser.add_argument(
        "--checkpoint", default=False, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-D",
        "--dataset",
        default="lfw",
        type=DatasetEnum,
        action=EnumAction,
        required=True,
    )
    parser.add_argument(
        "--instagram-filter", type=InstaFilterEnum, action=EnumAction, required=False
    )

    args = parser.parse_args()

    from json import load

    args.model_config = load(args.model_config)

    return args


def load_data():
    ds: DatasetEnum = hparams["dataset"]
    train_size, test_size = hparams["training"], hparams["testing"]

    # In the case that a number of samples is passed in instead of a percentage of the test split
    if train_size > 1.0:
        train_size = int(train_size)
    if test_size > 1.0:
        test_size = int(test_size)

    insta: InstaFilterEnum = hparams["instagram_filter"]
    data = ds.generate_ds(
        insta.name if insta else "default",
        hparams["batch"],
        insta.filter_transform if insta else None,
    )
    t_ds, v_ds = ds.get_loaders(
        data, hparams["input"], train_size, test_size, hparams["batch"]
    )

    # from keras import preprocessing

    # for example in t_ds:
    #     print(example[0])
    #     print(ops.max(example[0]), ops.min(example[0]))
    #     print(ops.max(example[1]), ops.min(example[1]))
    #     ex = ops.convert_to_numpy(example[0][0])
    #     ey = ops.convert_to_numpy(example[1][0])
    #     preprocessing.image.array_to_img(ex).save('train_in.png')
    #     preprocessing.image.array_to_img(ey).save('train_targ.png')
    #     break
    # for example in v_ds:
    #     print(example[0])
    #     print(ops.max(example[0]), ops.min(example[0]))
    #     print(ops.max(example[1]), ops.min(example[1]))
    #     ex = ops.convert_to_numpy(example[0][0])
    #     ey = ops.convert_to_numpy(example[1][0])
    #     preprocessing.image.array_to_img(ex).save('val_in.png')
    #     preprocessing.image.array_to_img(ey).save('val_targ.png')
    #     break
    # exit(1)

    hparams["dataset"] = ds.name.lower()

    return t_ds, v_ds


def initialize_loss():
    losses = []
    weights = []
    loss_config = {}
    cs_transforms = []
    metrics = []

    is_not_rgb = hparams["color_space"].name.casefold() != "rgb"
    F = hparams.pop("F")
    threshold = hparams.pop("threshold")
    rho = hparams.pop("rho")
    if F:
        for f in F:
            if threshold:  # Loss with thresholding
                losses.append(
                    FaceEmbeddingThresholdLoss(f=f, threshold=f.get_threshold())
                )
                weights.append(rho)
            else:  # Loss as described by ReFace
                losses.append(FaceEmbeddingLoss(f=f))
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
                tmp_loss = ls.MeanSquaredError()
                cs_transforms.append(False)
            elif loss_i == "mae":
                tmp_loss = ls.MeanAbsoluteError()
                cs_transforms.append(False)
            elif loss_i == "dsssim":
                tmp_loss = DSSIMObjective()
                cs_transforms.append(False)
            elif loss_i == "gsssim":
                tmp_loss = GRAYSSIMObjective()
                cs_transforms.append(False)
            elif loss_i == "nima":
                tmp_loss = AestheticLoss(name="NIMA-A", backbone="inceptionresnetv2")
                cs_transforms.append(is_not_rgb)
            elif loss_i == "exposure":
                tmp_loss = ExposureControlLoss(mean_val=0.6)
                cs_transforms.append(is_not_rgb)
            elif loss_i == "color":
                tmp_loss = ColorConstancyLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "illumination":
                tmp_loss = IlluminationSmoothnessLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "spatial":
                tmp_loss = SpatialConsistencyLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "style":
                style = hparams.pop("style_ref")
                tmp_loss = StyleLoss(reference=style)
                cs_transforms.append(is_not_rgb)
            elif loss_i == "variation":
                tmp_loss = VariationLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "content":
                tmp_loss = ContentLoss()
                cs_transforms.append(is_not_rgb)
            else:
                spatial = hparams.pop("lpips_spatial")
                tmp_loss = PerceptualLoss(
                    backbone=loss_i, spatial=spatial, tolerance=0.05
                )
                cs_transforms.append(is_not_rgb)

            losses.append(tmp_loss)
            weights.append(w_i)
            loss_config[tmp_loss.name] = tmp_loss.get_config() | {"weight": w_i}

    if not is_not_rgb:
        cs_transforms = None

    hparams["losses"] = loss_config

    return losses, weights, cs_transforms, metrics


def initialize_model():
    eps = hparams["epsilon"]
    base_model: BaseModels = hparams.pop("model_backbone")

    if base_model in [BaseModels.ZERODCE, BaseModels.RESZERODCE]:
        from auramask.models.zero_dce import get_enhanced_image

        postproc = get_enhanced_image

        def preproc(inputs):
            inputs = keras.layers.Rescaling(scale=2, offset=-1)(inputs)
            return inputs

    else:

        def preproc(inputs):
            inputs = keras.layers.Rescaling(scale=2, offset=-1)(inputs)
            return inputs

        def postproc(x: keras.KerasTensor, inputs: keras.KerasTensor):
            x = ops.multiply(eps, x)
            out = ops.add(x, inputs)
            out = ops.clip(out, 0.0, 1.0)
            return [out, x]

    model_config: dict = hparams["model_config"]

    # Allows modifying the config at calling with the AURAMASK_CONFIG environment variable
    cfg_mod: dict = ast.literal_eval(os.getenv("AURAMASK_CONFIG", "{}"))
    for key, val in cfg_mod.items():
        if isinstance(val, str) and val.lower() in ["true", "false"]:
            cfg_mod[key] = True if val.lower() == "true" else False
    model_config.update(cfg_mod)

    hparams["model"] = base_model.name.lower()
    model = base_model.build_backbone(
        model_config=model_config,
        input_shape=(224, 224, 3)
        if keras.backend.image_data_format() == "channels_last"
        else (3, 224, 224),
        preprocess=preproc,
        activation_fn=activations.tanh,
        post_processing=postproc,
        name=hparams["model"],
    )

    losses, losses_w, losses_t, metrics = initialize_loss()
    # schedule = opts.schedules.ExponentialDecay(
    #     initial_learning_rate=hparams["alpha"],
    #     decay_steps=1000,
    #     decay_rate=0.96,
    #     staircase=True,
    # )
    optimizer = opts.Adam(learning_rate=hparams["alpha"], clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=losses_w,
        loss_convert=losses_t,
        run_eagerly=hparams.pop("eager"),
        metrics=metrics,
        jit_compile=False,
    )

    return model


def set_seed():
    seed = hparams["seed"]
    seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % 10**8
    utils.set_random_seed(seed)


def get_sample_data(ds):
    if keras.backend.backend() == "tensorflow":
        for x in ds.take(1):
            inp = x[0][:8]
    else:
        for batch in ds:
            inp = batch[0][:8]
            break

    return inp


def main():
    # Constant Defaults
    hparams["optimizer"] = "adam"
    hparams["input"] = (256, 256)
    hparams.update(parse_args().__dict__)
    log = hparams.pop("log")
    logdir = hparams.pop("log_dir")
    note = hparams.pop("note")
    verbose = hparams.pop("verbose")
    mixed_precision = hparams.pop("mixed_precision")

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
    logdir = str(logdir)

    set_seed()
    # Load the training and validation data
    t_ds, v_ds = load_data()

    model = initialize_model()

    v = get_sample_data(v_ds)
    model(v)

    callbacks = init_callbacks(hparams, v, logdir, note)

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
