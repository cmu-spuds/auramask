# ruff: noqa: E402
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import wandb
from auramask.utils.backbones import BaseModels


def AuraMask(config: dict, weights: str):
    eps = config["epsilon"]
    base_model: BaseModels = BaseModels[config["model"].upper()]

    if base_model in [BaseModels.ZERODCE, BaseModels.RESZERODCE]:
        from auramask.models.zero_dce import get_enhanced_image

        postproc = get_enhanced_image
        preproc = None
    else:

        def preproc(inputs):
            inputs = keras.layers.Rescaling(scale=2, offset=-1)(inputs)
            return inputs

        def postproc(x: keras.KerasTensor, inputs: keras.KerasTensor):
            x = keras.ops.multiply(eps, x)
            out = keras.ops.add(x, inputs)
            out = keras.ops.clip(out, 0.0, 1.0)
            return [out, x]

    model_config: dict = config["model_config"]

    model = base_model.build_backbone(
        model_config=model_config,
        input_shape=(224, 224, 3)
        if keras.backend.image_data_format() == "channels_last"
        else (3, 224, 224),
        preprocess=preproc,
        activation_fn=keras.activations.tanh,
        post_processing=postproc,
        name="AuraMask",
    )

    model.build_from_config({"input_shape": (None, 224, 224, 3)})

    model.load_weights(weights)

    return model


if __name__ == "__main__":
    api = wandb.Api()
    artifact: wandb.Artifact = api.artifact(
        "spuds/auramask/lofi-reszero:latest", type="model"
    )
    weights = artifact.download()
    weights = os.path.join(weights, "101-0.43.weights.h5")
    run = artifact.logged_by()
    config = run.config
    model = AuraMask(config, weights)
    model.summary()
