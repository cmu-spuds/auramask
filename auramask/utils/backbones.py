from enum import Enum
from types import NoneType, FunctionType
from keras import Input, backend, layers
from keras.src.applications.imagenet_utils import obtain_input_shape
from keras_unet_collection import models as unet_models
from auramask.models import zero_dce, reface_unet, zero_dce_auramask, auramask


class BaseModels(Enum):
    UNET = (unet_models.unet_2d, False)
    R2UNET = (unet_models.r2_unet_2d, False)
    RESUNET = (unet_models.resunet_a_2d, False)
    ATTUNET = (unet_models.att_unet_2d, False)
    VNET = (unet_models.vnet_2d, False)
    ZERODCE = (zero_dce.build_dce_net, True)
    RESZERODCE = (zero_dce_auramask.build_res_dce_net, True)
    REFACE = (reface_unet.reface_unet, True)

    def build_backbone(
        self,
        model_config: dict,
        input_shape: tuple,
        name: str = None,
        preprocess: FunctionType | NoneType = None,
        activation_fn: FunctionType | str | NoneType = None,
        post_processing: FunctionType | NoneType = None,
    ):
        input_shape = obtain_input_shape(
            input_shape,
            default_size=224,
            min_size=112,
            data_format=backend.image_data_format(),
            require_flatten=False,
        )

        inputs = Input(shape=input_shape, name="{}_input".format(name))

        # Integrated preprocessing (e.g., color transform, scaling, normalizing)
        if preprocess:
            xx = preprocess(inputs)
        else:
            xx = inputs

        if self.value[1]:
            x = self.value[0](input_tensor=xx, **model_config).output[0]
        else:
            # Get model using config dict
            x = self.value[0](input_size=input_shape, **model_config)(xx)[0]

        # Use activation function if not defined by builder
        if activation_fn:
            x = layers.Activation(activation_fn)(x)

        # Integrated post-processing (e.g., scaling, mutiplying, adding to input, clipping)
        if post_processing:
            x = post_processing(x, inputs)

        # Use name of backbone if no custom name is given
        if not name:
            name = self.name.lower()

        # Create model for training
        model = auramask.AuraMask(inputs=inputs, outputs=x, name=name)

        return model
