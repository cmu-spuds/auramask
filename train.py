import argparse
import enum
import hashlib
import os
from pathlib import Path
from random import choice
from string import ascii_uppercase

from auramask.callbacks.callbacks import init_callbacks

from auramask.losses.ffl import FocalFrequencyLoss
from auramask.losses.perceptual import PerceptualLoss
from auramask.losses.embeddistance import (
    FaceEmbeddingThresholdLoss,
)
from auramask.losses.aesthetic import AestheticLoss
from auramask.losses.ssim import GRAYSSIM, MSSSIMLoss, SSIMLoss, YUVSSIMLoss
from auramask.losses.zero_dce import (
    ColorConstancyLoss,
    SpatialConsistencyLoss,
    ExposureControlLoss,
    IlluminationSmoothnessLoss,
)

from auramask.metrics.embeddistance import PercentageOverThreshold

from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.models.auramask import AuraMask

from auramask.utils.colorspace import ColorSpaceEnum
from auramask.utils.datasets import DatasetEnum

import tensorflow as tf

from keras.optimizers import Adam
from keras.losses import MeanSquaredError, MeanAbsoluteError

from datetime import datetime

from git import Repo

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

branch = Repo("./").active_branch.name  # Used for debugging runs

# Global hparams object
hparams: dict = {}


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
    parser.add_argument("-a", "--alpha", type=float, default=2e-4)
    parser.add_argument("-e", "--epsilon", type=float, default=0.03)
    parser.add_argument("-l", "--lambda", type=float, default=[1.0], nargs="+")
    parser.add_argument("-p", "--rho", type=float, default=1.0)
    parser.add_argument("-g", "--gamma", type=float, default=0.1)
    parser.add_argument("-B", "--batch-size", dest="batch", type=int, default=32)
    parser.add_argument("-E", "--epochs", type=int, default=5)
    parser.add_argument("-d", "--depth", type=int, default=5)
    parser.add_argument(
        "-L",
        "--lpips",
        type=str,
        default=["none"],
        choices=[
            "alex",
            "vgg",
            "squeeze",
            "mse",
            "mae",
            "ssim",
            "msssim",
            "gsssim",
            "nima",
            "ffl",
            "exposure",
            "color",
            "illumination",
            "spatial",
            "none",
        ],
        nargs="+",
    )
    parser.add_argument(
        "--aesthetic", default=False, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-F", type=FaceEmbedEnum, nargs="+", required=False, action=EnumAction
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
    parser.add_argument("--t-split", type=str, required=True)
    parser.add_argument("--v-split", type=str, required=True)
    parser.add_argument("--n-filters", type=int, default=64)
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

    args = parser.parse_args()

    return args


def load_data():
    ds: DatasetEnum = hparams["dataset"]
    t_ds, v_ds = ds.fetch_dataset(hparams["t_split"], hparams["v_split"])

    w, h = hparams["input"]

    augmenters = DatasetEnum.get_augmenters(
        {"augs_per_image": 1, "rate": 0.5},
        {"augs_per_image": 1, "rate": 0.2, "magnitude": 0.5},
    )

    t_ds = (
        t_ds.to_tf_dataset(
            columns=ds.value[2],
            batch_size=hparams["batch"],
            collate_fn=DatasetEnum.data_collater,
            collate_fn_args={"args": {"w": w, "h": h, "crop": True}},
            prefetch=True,
            shuffle=True,
        )
        .cache()
        .map(
            lambda x: DatasetEnum.data_augmenter(
                x, augmenters["geom"], augmenters["aug"]
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    v_ds = v_ds.to_tf_dataset(
        columns=ds.value[2],
        batch_size=hparams["batch"],
        collate_fn=DatasetEnum.data_collater,
        collate_fn_args={"args": {"w": w, "h": h, "crop": True}},
        prefetch=True,
    ).cache()

    # import tensorflow_datasets as tfds

    # for example in v_ds.take(1):
    #     print(example)
    #     print(tf.reduce_mean(example, axis=[-1,-2,-3]), tf.math.reduce_std(example, axis=[-1,-2,-3]))
    #     print(tf.reduce_max(example, axis=[-1,-2,-3]), tf.reduce_min(example, axis=[-1,-2,-3]))

    # tfds.benchmark(v_ds)
    # tfds.benchmark(t_ds)

    # exit()

    return t_ds, v_ds


def initialize_loss():
    losses = []
    weights = []
    loss_config = []
    cs_transforms = []
    metrics = []

    is_not_rgb = hparams["color_space"].name.casefold() != "rgb"
    F = hparams.pop("F")
    rho = hparams.pop("rho")
    if F:
        for f in F:
            losses.append(FaceEmbeddingThresholdLoss(f=f, threshold=f.get_threshold()))
            weights.append(rho)
            # metrics.append(CosineDistance(f=losses[-1].f))
            metrics.append(
                PercentageOverThreshold(f=losses[-1].f, threshold=f.get_threshold())
            )
            loss_config.append(losses[-1].get_config() | {"weight": weights[-1]})
            cs_transforms.append(
                is_not_rgb
            )  # Determine if it needs to be transformed to rgb space

    if hparams.pop("aesthetic"):
        losses.append(
            AestheticLoss(name="NIMA-A", kind="nima-aes", backbone="nasnetmobile")
        )
        weights.append(hparams.pop("gamma"))
        loss_config.append(losses[-1].get_config() | {"weight": weights[-1]})
        cs_transforms.append(is_not_rgb)

    if "none" not in hparams["lpips"]:
        lam = hparams.pop("lambda")
        lpips = set(hparams.pop("lpips"))
        if len(lpips) != len(lam) and len(lam) > 1:
            raise argparse.ArgumentError(
                "The length of lambda values must equal that of lpips argument"
            )
        elif len(lam) <= 1:
            w = lam[0] if len(lam) > 0 else 1.0
            iters = zip(lpips, [w] * len(lpips))
        else:
            iters = zip(lpips, lam)

        for loss_i, w_i in iters:
            if loss_i == "mse":
                tmp_loss = MeanSquaredError()
                cs_transforms.append(False)
            elif loss_i == "mae":
                tmp_loss = MeanAbsoluteError()
                cs_transforms.append(False)
            elif loss_i == "ssim":
                tmp_loss = (
                    SSIMLoss(
                        max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
                    )
                    if hparams["color_space"].name.casefold() != "yuv"
                    else YUVSSIMLoss()
                )
                cs_transforms.append(False)
            elif loss_i == "gsssim":
                tmp_loss = GRAYSSIM(
                    max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
                )
                cs_transforms.append(is_not_rgb)
            elif loss_i == "msssim":
                tmp_loss = MSSSIMLoss()
                cs_transforms.append(False)
            elif loss_i == "ffl":
                tmp_loss = FocalFrequencyLoss()
                cs_transforms.append(False)
            elif loss_i == "nima":
                tmp_loss = AestheticLoss(name="NIMA-T", kind="nima-tech")
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
            else:
                tmp_loss = PerceptualLoss(backbone=loss_i)
                cs_transforms.append(is_not_rgb)

            losses.append(tmp_loss)
            weights.append(w_i)
            loss_config.append(tmp_loss.get_config() | {"weight": w_i})

    if not is_not_rgb:
        cs_transforms = None

    hparams["losses"] = loss_config

    return losses, weights, cs_transforms, metrics


def initialize_model():
    model = AuraMask(
        n_filters=hparams["n_filters"],
        n_dims=3,
        eps=hparams["epsilon"],
        depth=hparams["depth"],
        colorspace=hparams["color_space"].value,
    )

    hparams["model"] = model.model.name

    losses, losses_w, losses_t, metrics = initialize_loss()
    optimizer = Adam(learning_rate=hparams["alpha"])
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=losses_w,
        loss_convert=losses_t,
        run_eagerly=hparams["eager"],
        metrics=metrics,
    )
    return model


def set_seed():
    from keras.utils import set_random_seed

    seed = hparams["seed"]
    seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % 10**8
    set_random_seed(seed)
    hparams["seed"] = seed


def get_sample_data(ds):
    for x in ds.take(1):
        inp = x[:8]
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

    # Load the training and validation data
    t_ds, v_ds = load_data()

    # Initialize the model with the input hyperparameters
    model = initialize_model()

    if log:
        if note:
            note = input("Note for Run:")
        else:
            note = ""
        if not logdir:
            logdir = Path(
                os.path.join(
                    "logs",
                    branch,
                    datetime.now().strftime("%m-%d"),
                    datetime.now().strftime("%H.%M"),
                )
            )
        else:
            logdir = Path(
                os.path.join(
                    logdir,
                    datetime.now().strftime("%m-%d"),
                    datetime.now().strftime("%H.%M"),
                )
            )
        logdir.mkdir(parents=True, exist_ok=True)
        logdir = str(logdir)
        v = get_sample_data(v_ds)
        # t = get_sample_data(t_ds)
        # model(v[0])
        callbacks = init_callbacks(hparams, v, logdir, note)
    else:
        callbacks = None

    set_seed()

    training_history = model.fit(
        t_ds,
        callbacks=callbacks,
        epochs=hparams["epochs"],
        verbose=verbose,
        validation_data=v_ds,
    )
    return training_history


if __name__ == "__main__":
    main()
