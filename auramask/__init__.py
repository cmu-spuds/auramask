import keras
from typing import Optional
from auramask.utils import constants
from auramask.utils import pcgrad  # noqa: F401
from auramask import callbacks, losses, metrics  # noqa: F401
from auramask.models.zero_dce import get_enhanced_image


def AuraMask(config: dict, weights: Optional[str] = None):
    eps = config["epsilon"]
    base_model: constants.BaseModels = constants.BaseModels[config["model"].upper()]
    model_config: dict = config["model_config"]

    activation_fn = keras.activations.tanh

    if base_model in [constants.BaseModels.ZERODCE, constants.BaseModels.RESZERODCE]:
        postproc = get_enhanced_image
        preproc = None
    else:

        def preproc(inputs):
            inputs = keras.layers.Rescaling(scale=2, offset=-1)(inputs)
            return inputs

        if model_config["n_labels"] == 24:
            postproc = get_enhanced_image
        elif eps < 1:

            def postproc(x: keras.KerasTensor, inputs: keras.KerasTensor):
                x = keras.ops.multiply(eps, x)
                out = keras.ops.add(x, inputs)
                out = keras.ops.clip(out, 0.0, 1.0)
                return [out, x]
        else:

            def postproc(x: keras.KerasTensor, inputs: keras.KerasTensor):
                return [x, keras.ops.subtract(inputs, x)]

            activation_fn = keras.activations.sigmoid

    model = base_model.build_backbone(
        model_config=model_config,
        input_shape=(224, 224, 3)
        if keras.backend.image_data_format() == "channels_last"
        else (3, 224, 224),
        preprocess=preproc,
        activation_fn=activation_fn,
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
