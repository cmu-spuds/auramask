import keras
from torch import NoneType
from auramask.utils import constants
from auramask import callbacks, losses  # noqa: F401


def AuraMask(config: dict, weights: str | NoneType = None):
    eps = config["epsilon"]
    base_model: constants.BaseModels = constants.BaseModels[config["model"].upper()]

    if base_model in [constants.BaseModels.ZERODCE, constants.BaseModels.RESZERODCE]:
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

    if keras.backend.image_data_format() == "channels_last":
        model.build_from_config({"input_shape": (None, 224, 224, 3)})
    else:
        model.build_from_config({"input_shape": (None, 3, 224, 224)})

    if weights:
        model.load_weights(weights)

    return model
