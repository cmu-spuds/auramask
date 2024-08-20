from typing import Literal
from keras import layers, Model, applications, utils


def NIMA(
    backbone: Literal["mobilenet"]
    | Literal["nasnetmobile"]
    | Literal["inceptionresnetv2"],
    rescale: bool = True,
):
    """_summary_

    Args:
        backbone: Right now only "mobilenet"
    """
    inp = layers.Input((None, None, 3))

    if rescale:
        inp = layers.Rescaling(scale=255, offset=0)(inp)

    if backbone == "inceptionresnetv2":
        x = applications.inception_resnet_v2.preprocess_input(inp)
        base_model = applications.InceptionResNetV2(
            input_tensor=x, include_top=False, pooling="avg", weights=None
        )
        weight_path = utils.get_file(
            fname="nima_resnetv2.h5",
            origin="https://github.com/titu1994/neural-image-assessment/releases/download/v0.5/inception_resnet_weights.h5",
            cache_subdir="weights",
        )
    elif backbone == "mobilenet":
        x = applications.mobilenet.preprocess_input(inp)
        base_model = applications.mobilenet.MobileNet(
            input_tensor=x, include_top=False, alpha=1, pooling="avg", weights=None
        )
        weight_path = utils.get_file(
            fname="nima_mobilenet.h5",
            origin="https://github.com/titu1994/neural-image-assessment/releases/download/v0.3/mobilenet_weights.h5",
            cache_subdir="weights",
        )
    elif backbone == "nasnetmobile":
        x = layers.Resizing(224, 224)(inp)
        x = applications.nasnet.preprocess_input(x)
        base_model = applications.nasnet.NASNetMobile(
            input_tensor=x, include_top=False, pooling="avg", weights=None
        )
        weight_path = utils.get_file(
            fname="nima_nasnet.h5",
            origin="https://github.com/titu1994/neural-image-assessment/releases/download/v0.4/nasnet_weights.h5",
            cache_subdir="weights",
        )
    else:
        raise ValueError("Provided invalid backbone option %s", backbone)

    x = layers.Dropout(0.75)(base_model.output)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(inp, x, name="NIMA-%s" % (backbone))
    model.load_weights(weight_path)
    model.trainable = False

    return model
