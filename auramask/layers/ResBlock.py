from keras import (
    layers,
    activations,
    regularizers,
    constraints,
    initializers,
    saving,
    backend as K,
)
from typing import Tuple, List, Union, AnyStr, Callable, Dict, Optional, Type
from keras.src.utils.argument_validation import standardize_padding, standardize_tuple
from keras.src.backend import standardize_data_format
from keras.src.ops.operation_utils import compute_conv_output_shape
from keras.src.backend.common.backend_utils import compute_conv_transpose_output_shape
from auramask.layers import Conv1DTranspose

ConvND = Union[layers.Conv1D, layers.Conv2D, layers.Conv3D]


def get_conv_layer_type(rank: int) -> Type[ConvND]:
    if rank not in (1, 2, 3):
        raise ValueError("Rank must either be 1, 2 or 3. Received {}.".format(rank))
    conv_layer_types = [None, layers.Conv1D, layers.Conv2D, layers.Conv3D]
    return conv_layer_types[rank]


# region Residual scalars (Multiplier/Bias)
@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResidualMultiplier(layers.Layer):
    def __init__(self, initial_value: float, **kwargs):
        super(ResidualMultiplier, self).__init__(**kwargs)
        self.initial_value = initial_value
        self.multiplier = None

    def build(self, input_shape):
        self.multiplier = self.add_weight(
            name="multiplier",
            shape=[],
            dtype=K.floatx(),
            initializer=initializers.Constant(value=self.initial_value),
        )

    def call(self, inputs, **kwargs):
        return inputs * self.multiplier

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        base_config = super(ResidualMultiplier, self).get_config()
        config = {
            **base_config,
            "initial_value": self.initial_value,
        }
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResidualBias(layers.Layer):
    def __init__(self, **kwargs):
        super(ResidualBias, self).__init__(**kwargs)
        self.bias = None

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias", shape=[], dtype=K.floatx(), initializer=initializers.Zeros()
        )

    def call(self, inputs, **kwargs):
        return inputs + self.bias

    def compute_output_signature(self, input_signature):
        pass


# endregion


def normalize_padding(padding: str) -> str:
    padding = standardize_padding(padding)
    if padding == "valid":
        raise ValueError(
            "`valid` padding is not currently supported for residual blocks."
        )
    return padding


# region Basic blocks
@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBasicBlockND(layers.Layer):
    def __init__(
        self,
        rank: int,
        filters: int,
        depth: int,
        kernel_size: Union[int, Tuple, List],
        strides: Union[int, Tuple, List],
        padding: str,
        data_format: Optional[AnyStr],
        dilation_rate: Union[int, Tuple, List],
        kernel_regularizer: Optional[Union[Dict, AnyStr, Callable]],
        bias_regularizer: Optional[Union[Dict, AnyStr, Callable]],
        activity_regularizer: Optional[Union[Dict, AnyStr, Callable]],
        kernel_constraint: Optional[Union[Dict, AnyStr, Callable]],
        bias_constraint: Optional[Union[Dict, AnyStr, Callable]],
        **kwargs,
    ):
        # region Check parameters
        if rank not in (1, 2, 3):
            raise ValueError("Rank must either be 1, 2 or 3. Received {}.".format(rank))

        if depth <= 0:
            raise ValueError(
                "Depth must be strictly positive. Received {}.".format(depth)
            )
        # endregion

        super(ResBasicBlockND, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.depth = depth

        self.kernel_size = standardize_tuple(kernel_size, rank, "kernel_size")
        self.strides = standardize_tuple(strides, rank, "strides")
        self.padding = normalize_padding(padding)

        self.data_format = standardize_data_format(data_format)
        self.dilation_rate = standardize_tuple(dilation_rate, rank, "dilation_rate")

        self.kernel_initializer = initializers.VarianceScaling(mode="fan_in")
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.conv_layers: List[layers.Layer] = []
        self.projection_layer: Optional[layers.Layer] = None
        self.residual_multiplier = None

        self.input_spec = layers.InputSpec(ndim=self.rank + 2)
        self.init_layers()

    def init_layer(self, use_strides: bool, projection_layer: bool) -> ConvND:
        strides = self.strides if (use_strides or projection_layer) else 1
        kernel_size = (
            self.projection_kernel_size if projection_layer else self.kernel_size
        )
        return self.conv_layer_type(
            filters=self.filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            use_bias=not projection_layer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer="zeros",
            kernel_regularizer=self.kernel_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

    def init_layers(self):
        for i in range(self.depth):
            conv_layer = self.init_layer(i == 0, projection_layer=False)
            self.conv_layers.append(conv_layer)

        self.residual_multiplier = ResidualMultiplier(initial_value=0.0)

    def build(self, input_shape):
        if self.use_projection(input_shape):
            self.init_projection_layer()

        self.input_spec = layers.InputSpec(
            ndim=self.rank + 2, axes={self.channel_axis: input_shape[self.channel_axis]}
        )
        super(ResBasicBlockND, self).build(input_shape)

    def init_projection_layer(self):
        self.projection_layer = self.init_layer(use_strides=True, projection_layer=True)

    def call(self, inputs, **kwargs):
        outputs = inputs

        for i in range(self.depth):
            outputs = activations.relu(outputs)
            outputs = self.conv_layers[i](outputs)

        outputs = self.residual_multiplier(outputs)

        if self.projection_layer is not None:
            inputs = self.projection_layer(inputs)

        outputs = inputs + outputs

        return outputs

    def use_projection(self, input_shape):
        strides = str(self.strides)
        for stride in strides:
            if stride != 1:
                return True

        return input_shape[self.channel_axis] != self.filters

    # noinspection DuplicatedCode
    def compute_output_shape(self, input_shape):
        def get_new_space(space):
            new_space = []
            for i in range(len(space)):
                new_dim = compute_conv_output_shape(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tuple(new_space)

        if self.channels_first:
            return (input_shape[0], self.filters) + get_new_space(input_shape[2:])
        else:
            return (
                (input_shape[0],) + get_new_space(input_shape[1:-1]) + (self.filters,)
            )

    @property
    def conv_layer_type(self) -> Type[ConvND]:
        return get_conv_layer_type(self.rank)

    @property
    def channel_axis(self):
        if self.data_format == "channels_first":
            return 1
        else:
            return -1

    @property
    def channels_first(self):
        return self.data_format == "channels_first"

    @property
    def projection_kernel_size(self):
        return standardize_tuple(1, self.rank, "projection_kernel_size")

    def get_config(self):
        config = {
            "rank": self.rank,
            "filters": self.filters,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(ResBasicBlockND, self).get_config()
        return {**base_config, **config}

    def compute_output_signature(self, input_signature):
        pass


# region ResBasicBlock - 1D/2D/3D
@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBasicBlock1D(ResBasicBlockND):
    def __init__(
        self,
        filters,
        depth=2,
        kernel_size=3,
        strides=1,
        padding="same",
        data_format=None,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBasicBlock1D, self).__init__(
            rank=1,
            filters=filters,
            depth=depth,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBasicBlock1D, self).get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBasicBlock2D(ResBasicBlockND):
    def __init__(
        self,
        filters,
        depth=2,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format=None,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBasicBlock2D, self).__init__(
            rank=2,
            filters=filters,
            depth=depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBasicBlock2D, self).get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBasicBlock3D(ResBasicBlockND):
    def __init__(
        self,
        filters,
        depth=2,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        data_format=None,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBasicBlock3D, self).__init__(
            rank=3,
            filters=filters,
            depth=depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBasicBlock3D, self).get_config()
        config.pop("rank")
        return config


# endregion


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBasicBlockNDTranspose(ResBasicBlockND):
    @property
    def conv_layer_type(self) -> Type[ConvND]:
        conv_layer_types = [
            None,
            Conv1DTranspose,
            layers.Conv2DTranspose,
            layers.Conv2DTranspose,
        ]
        return conv_layer_types[self.rank]

    # noinspection DuplicatedCode
    def compute_output_shape(self, input_shape):
        return compute_conv_transpose_output_shape(
            input_shape=input_shape,
            kernel_size=self.kernel_size,
            filters=self.filters,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )


# region ResBasicBlock - 1D/2D/3D
@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBasicBlock1DTranspose(ResBasicBlockNDTranspose):
    def __init__(
        self,
        filters: int,
        depth=2,
        kernel_size=3,
        strides=1,
        padding="same",
        data_format=None,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBasicBlock1DTranspose, self).__init__(
            rank=1,
            filters=filters,
            depth=depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            dilation_rate=1,
            **kwargs,
        )
        self._padding = padding

    def get_config(self):
        config = super(ResBasicBlock1DTranspose, self).get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBasicBlock2DTranspose(ResBasicBlockNDTranspose):
    def __init__(
        self,
        filters: int,
        depth=2,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format=None,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBasicBlock2DTranspose, self).__init__(
            rank=2,
            filters=filters,
            depth=depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            dilation_rate=1,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBasicBlock2DTranspose, self).get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBasicBlock3DTranspose(ResBasicBlockNDTranspose):
    def __init__(
        self,
        filters: int,
        depth=2,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        data_format=None,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBasicBlock3DTranspose, self).__init__(
            rank=3,
            filters=filters,
            depth=depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            dilation_rate=1,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBasicBlock3DTranspose, self).get_config()
        config.pop("rank")
        return config


# endregion
# endregion

# region ResBlocks


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBlockND(layers.Layer):
    def __init__(
        self,
        rank: int,
        filters: int,
        basic_block_count=1,
        basic_block_depth=1,
        kernel_size: Union[int, Tuple, List] = 3,
        strides: Union[int, Tuple, List] = 1,
        padding="same",
        data_format: AnyStr = None,
        dilation_rate: Union[int, Tuple, List] = 1,
        activation: Union[None, AnyStr, Callable] = "linear",
        kernel_regularizer: Union[Dict, AnyStr, Callable] = None,
        bias_regularizer: Union[Dict, AnyStr, Callable] = None,
        activity_regularizer: Union[Dict, AnyStr, Callable] = None,
        kernel_constraint: Union[Dict, AnyStr, Callable] = None,
        bias_constraint: Union[Dict, AnyStr, Callable] = None,
        **kwargs,
    ):
        if rank not in [1, 2, 3]:
            raise ValueError("`rank` must be in [1, 2, 3]. Got {}".format(rank))

        if not isinstance(basic_block_count, int) or (basic_block_count <= 0):
            raise ValueError(
                "`basic_block_count` must be a strictly positive integer. Got {}".format(
                    basic_block_count
                )
            )

        super(ResBlockND, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.basic_block_count = basic_block_count
        self.basic_block_depth = basic_block_depth

        self.kernel_size = standardize_tuple(kernel_size, rank, "kernel_size")
        self.strides = standardize_tuple(strides, rank, "strides")
        self.padding = normalize_padding(padding)

        self.data_format = standardize_data_format(data_format)
        self.dilation_rate = standardize_tuple(dilation_rate, rank, "dilation_rate")
        self.activation = activations.get(activation)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.basic_blocks: List[ResBasicBlockND] = []

        self.input_spec = layers.InputSpec(ndim=self.rank + 2)
        self.init_layers()

    def init_layers(self):
        for i in range(self.basic_block_count):
            strides = self.strides if (i == 0) else 1
            basic_block = ResBasicBlockND(
                rank=self.rank,
                filters=self.filters,
                depth=self.basic_block_depth,
                kernel_size=self.kernel_size,
                strides=strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
            )
            self.basic_blocks.append(basic_block)

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(
            ndim=self.rank + 2, axes={self.channel_axis: input_shape[self.channel_axis]}
        )
        super(ResBlockND, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = inputs
        for basic_block in self.basic_blocks:
            outputs = basic_block(outputs)
        outputs = self.activation(outputs)
        return outputs

    # noinspection DuplicatedCode
    def compute_output_shape(self, input_shape):
        return compute_conv_output_shape(
            input_shape,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    @property
    def channel_axis(self):
        if self.data_format == "channels_first":
            return 1
        else:
            return -1

    @property
    def channels_first(self):
        return self.data_format == "channels_first"

    def get_config(self):
        if isinstance(self.activation, layers.Activation):
            activation = self.activation.get_config()
        else:
            activation = activations.serialize(self.activation)

        config = {
            "rank": self.rank,
            "filters": self.filters,
            "basic_block_count": self.basic_block_count,
            "basic_block_depth": self.basic_block_depth,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activation,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }

        base_config = super(ResBlockND, self).get_config()
        return {**base_config, **config}

    def compute_output_signature(self, input_signature):
        pass


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBlock1D(ResBlockND):
    def __init__(
        self,
        filters,
        basic_block_count=1,
        basic_block_depth=1,
        kernel_size=3,
        strides=1,
        padding="same",
        data_format=None,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBlock1D, self).__init__(
            rank=1,
            filters=filters,
            basic_block_count=basic_block_count,
            basic_block_depth=basic_block_depth,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBlock1D, self).get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBlock2D(ResBlockND):
    def __init__(
        self,
        filters,
        basic_block_count=1,
        basic_block_depth=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format=None,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBlock2D, self).__init__(
            rank=2,
            filters=filters,
            basic_block_count=basic_block_count,
            basic_block_depth=basic_block_depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBlock2D, self).get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBlock3D(ResBlockND):
    def __init__(
        self,
        filters,
        basic_block_count=1,
        basic_block_depth=1,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        data_format=None,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBlock3D, self).__init__(
            rank=3,
            filters=filters,
            basic_block_count=basic_block_count,
            basic_block_depth=basic_block_depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBlock3D, self).get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBlockNDTranspose(ResBlockND):
    def init_layers(self):
        for i in range(self.basic_block_count):
            strides = self.strides if (i == 0) else 1
            basic_block = ResBasicBlockNDTranspose(
                rank=self.rank,
                filters=self.filters,
                depth=self.basic_block_depth,
                kernel_size=self.kernel_size,
                strides=strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
            )
            self.basic_blocks.append(basic_block)

    # noinspection DuplicatedCode
    def compute_output_shape(self, input_shape):
        return compute_conv_transpose_output_shape(
            input_shape=input_shape,
            kernel_size=self.kernel_size,
            filters=self.filters,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBlock1DTranspose(ResBlockNDTranspose):
    def __init__(
        self,
        filters,
        basic_block_count=1,
        basic_block_depth=1,
        kernel_size=3,
        strides=1,
        data_format=None,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBlock1DTranspose, self).__init__(
            rank=1,
            filters=filters,
            basic_block_count=basic_block_count,
            basic_block_depth=basic_block_depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBlock1DTranspose, self).get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBlock2DTranspose(ResBlockNDTranspose):
    def __init__(
        self,
        filters,
        basic_block_count=1,
        basic_block_depth=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format=None,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBlock2DTranspose, self).__init__(
            rank=2,
            filters=filters,
            basic_block_count=basic_block_count,
            basic_block_depth=basic_block_depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBlock2DTranspose, self).get_config()
        config.pop("rank")
        return config


@saving.register_keras_serializable("auramask.layers.ResBlock")
class ResBlock3DTranspose(ResBlockNDTranspose):
    def __init__(
        self,
        filters,
        basic_block_count=1,
        basic_block_depth=1,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        data_format=None,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ResBlock3DTranspose, self).__init__(
            rank=3,
            filters=filters,
            basic_block_count=basic_block_count,
            basic_block_depth=basic_block_depth,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super(ResBlock3DTranspose, self).get_config()
        config.pop("rank")
        return config


# endregion
