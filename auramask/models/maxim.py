from types import NoneType
from typing import Literal
from keras import layers, backend, utils, Model, initializers
from huggingface_hub import from_pretrained_keras

BASE_URL = "google/maxim"


def Maxim(
    weights: str | Literal["hf"] | NoneType = "hf",
    arch: Literal["s2"] | Literal["s3"] | NoneType = "s2",
    task: Literal["denoising"]
    | Literal["enhancement"]
    | Literal["deblurring"]
    | Literal["deraining"]
    | Literal["dehazing"]
    | NoneType = "enhancement",
    dataset: Literal["lol"]
    | Literal["fivek"]
    | Literal["sots-indoor"]
    | Literal["rain13k"]
    | Literal["realblur-r"]
    | Literal["gopro"]
    | Literal["reds"]
    | Literal["sidd"]
    | Literal["raindrop"]
    | Literal["realblur-j"]
    | Literal["sots-outdoor"]
    | NoneType = "fivek",
    input_tensor=None,
    input_shape=None,
    name="Maxim",
):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if "hf" in weights:
        path = "%s-%s-%s-%s" % (BASE_URL, arch, task, dataset)
        name = "%s-%s" % (name, path)
        x = from_pretrained_keras(path)(img_input)
    elif weights is None:
        mdl = from_pretrained_keras("google/maxim-s2-enhancement-fivek")
        with initializers.GlorotUniform() as init:
            for w in mdl.trainable_variables:
                w.assign(init(w.shape))
        x = mdl(x)
    else:
        mdl = from_pretrained_keras("google/maxim-s2-enhancement-fivek")
        mdl.load_weights(weights)
        x = mdl(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    return model
