# ruff: noqa: E402
import argparse
import enum
import hashlib
import os
import ast
from pathlib import Path
from random import choice
from string import ascii_uppercase
from datetime import datetime

os.environ["KERAS_BACKEND"] = "torch"

import keras

from auramask.callbacks.callbacks import init_callbacks

from auramask.losses.content import ContentLoss
from auramask.losses.perceptual import PerceptualLoss
from auramask.losses.embeddistance import (
    FaceEmbeddingLoss,
    FaceEmbeddingThresholdLoss,
)
from auramask.losses.aesthetic import AestheticLoss
from auramask.losses.ssim import DSSIMObjective, GRAYSSIMObjective
from auramask.losses.style import StyleLoss, StyleRefs
from auramask.losses.variation import VariationLoss
from auramask.losses.zero_dce import (
    ColorConstancyLoss,
    SpatialConsistencyLoss,
    ExposureControlLoss,
    IlluminationSmoothnessLoss,
)


from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.models.zero_dce import get_enhanced_image

from auramask.utils import backbones
from auramask.utils.insta_filter import InstaFilterEnum
from auramask.utils.colorspace import ColorSpaceEnum
from auramask.utils.datasets import DatasetEnum

from keras import optimizers as opts, losses as ls, activations, ops, utils

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Global hparams object
hparams: dict = {}
# keras.config.disable_traceback_filtering()


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


# Action for enumeration input
class EnumAction(argparse.Action):
    """Action for an enumeration input, maps enumeration type to choices"""

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name.lower() for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        if isinstance(values, str):
            value = self._enum[values.upper()]
        elif isinstance(values, list):
            value = [self._enum[x.upper()] for x in values]
        setattr(namespace, self.dest, value)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="AuraMask Training",
        description="A training script for the AuraMask network",
    )
    parser.add_argument(
        "-m",
        "--model-backbone",
        type=backbones.BaseModels,
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
    parser.add_argument("--training", type=str, required=True)
    parser.add_argument("--validation", type=str, required=True)
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
    t_ds, v_ds = ds.fetch_dataset(hparams["training"], hparams["validation"])

    w, h = hparams["input"]

    augmenters = ds.get_augmenters(
        {"augs_per_image": 1, "rate": 0.5},
        {"augs_per_image": 1, "rate": 0.2, "magnitude": 0.5},
    )

    if keras.backend.backend() == "tensorflow":
        keras.backend.set_image_data_format("channels_last")
        t_ds = (
            t_ds.to_tf_dataset(
                columns=ds.value[2],
                batch_size=hparams["batch"],
                collate_fn=ds.data_collater,
                collate_fn_args={"args": {"w": w, "h": h}},
                prefetch=False,
                shuffle=True,
                drop_remainder=True,
            )
            .map(
                lambda x: ds.data_augmenter(x, augmenters["geom"], augmenters["aug"]),
                num_parallel_calls=-1,
            )
            .repeat()
            .prefetch(-1)
        )

        v_ds = (
            v_ds.to_tf_dataset(
                columns=ds.value[2],
                batch_size=hparams["batch"],
                collate_fn=ds.data_collater,
                collate_fn_args={"args": {"w": w, "h": h}},
                prefetch=True,
                drop_remainder=True,
            )
            .cache()
            .prefetch(-1)
        )

    elif keras.backend.backend() == "torch":
        keras.backend.set_image_data_format("channels_last")
        insta = hparams["instagram_filter"]
        from torch.utils.data import DataLoader

        # This transform collates the data, converting all features into tensors, scaling to 0-1, resizing, and cropping
        # Then it augments the data according to the random geometric and pixel-level augmentations set in preprocessing
        # Finally, the embeddings of the unaltered image is pre-computed for each of the models in F
        def transform(example):
            example = ds.data_collater(example, {"w": w, "h": h})
            return ds.data_augmenter(example, augmenters["geom"], augmenters["aug"])

        if insta:
            t_ds = t_ds.map(
                lambda x: insta.filter_transform(x),
                input_columns=ds.value[-1][-1],
                batched=True,
                batch_size=32,
                num_proc=16,
            ).select_columns(ds.value[-1] + ["target"])
        else:
            t_ds = t_ds.select_columns(ds.value[-1])
        t_ds = DataLoader(
            t_ds,
            int(hparams["batch"]),
            shuffle=True,
            drop_last=True,
            collate_fn=transform,
        )

        def v_transform(example):
            example = ds.data_collater(example, {"w": w, "h": h})
            x = ops.convert_to_tensor(example[ds.value[-1][0]])
            if len(ds.value[-1]) > 1:
                y = ops.convert_to_tensor(example[ds.value[-1][1]])
            elif "target" in example.keys():
                y = ops.convert_to_tensor(example["target"])
            else:
                y = ops.copy(x)
            return (x, y)

        if insta:
            v_ds = v_ds.map(
                lambda x: insta.filter_transform(x),
                input_columns=ds.value[-1][-1],
                batched=True,
                batch_size=32,
                num_proc=16,
            )
        else:
            v_ds = v_ds.select_columns(ds.value[-1])
        v_ds = DataLoader(
            v_ds, int(hparams["batch"]), shuffle=False, collate_fn=v_transform
        )

    # from keras import preprocessing

    # for example in t_ds:
    #     ex = ops.convert_to_numpy(example[0])
    #     ey = ops.convert_to_numpy(example[1][0])
    #     preprocessing.image.array_to_img(ex[3]).save('train_in.png')
    #     preprocessing.image.array_to_img(ey[3]).save('train_targ.png')
    #     break
    # for example in v_ds:
    #     ex = ops.convert_to_numpy(example[0])
    #     ey = ops.convert_to_numpy(example[1][0])
    #     preprocessing.image.array_to_img(ex[0]).save('val_in.png')
    #     preprocessing.image.array_to_img(ey[0]).save('val_targ.png')
    #     break
    # # exit(1)

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
                    backbone=loss_i, spatial=spatial, tolerance=0.2
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
    base_model: backbones.BaseModels = hparams.pop("model_backbone")

    if base_model in [backbones.BaseModels.ZERODCE, backbones.BaseModels.RESZERODCE]:
        postproc = get_enhanced_image
        preproc = None
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
    schedule = opts.schedules.ExponentialDecay(
        initial_learning_rate=hparams["alpha"],
        decay_steps=500,
        decay_rate=0.96,
        staircase=True,
    )
    optimizer = opts.Adam(learning_rate=schedule, clipnorm=1.0)
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
