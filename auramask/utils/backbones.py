from enum import Enum
from functools import partial
from types import NoneType, FunctionType
from keras import Model, Input, backend
from keras_unet_collection import models as unet_models
from auramask.models import zero_dce, reface_unet, zero_dce_auramask

input_shape = (
    (None, None, 3)
    if backend.image_data_format() == "channels_last"
    else (3, None, None)
)


class BaseModels(Enum):
    UNET = partial(unet_models.unet_2d, input_size=input_shape)
    R2UNET = partial(unet_models.r2_unet_2d, input_size=input_shape)
    ATTUNET = partial(unet_models.att_unet_2d, input_size=input_shape)
    ZERODCE = partial(zero_dce.build_dce_net, input_shape=input_shape)
    RESZERODCE = partial(zero_dce_auramask.build_res_dce_net, input_shape=input_shape)
    REFACE = partial(reface_unet.reface_unet, input_shape=input_shape)

    def build_backbone(
        self,
        model_config: dict,
        name: str = None,
        preprocess: FunctionType | NoneType = None,
        activation_fn: FunctionType | NoneType = None,
        post_processing: FunctionType | NoneType = None,
    ):
        inputs = Input(shape=input_shape)

        # Integrated preprocessing (e.g., color transform, scaling, normalizing)
        if preprocess:
            inputs = preprocess(inputs)

        # Get model using config dict
        x = self.value(**model_config)(inputs)[0]

        # Use activation function if not defined by builder
        if activation_fn:
            x = activation_fn(x)

        # Integrated post-processing (e.g., scaling, mutiplying, adding to input, clipping)
        if post_processing:
            x = post_processing(x, inputs)

        # Use name of backbone if no custom name is given
        if not name:
            name = self.name.lower()

        # Create model for training
        backbone: Model = Model(inputs=inputs, outputs=x, name=name)

        return backbone
