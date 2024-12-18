from keras import Loss, ops, backend


class TopIQFR(Loss):
    def __init__(
        self,
        name="TopIQ",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if backend.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        import pyiqa

        self.model = pyiqa.create_metric("topiq_fr", as_loss=True)

    def get_config(self):
        base_config = super().get_config()
        config = {"full_ref": True, "lower_better": self.model.lower_better}
        return {**base_config, **config}

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        # Library only supports channels first so change incoming data
        if backend.image_data_format() == "channels_last":
            y_true = ops.moveaxis(y_true, -1, 1)
            y_pred = ops.moveaxis(y_pred, -1, 1)
        return 1 - self.model(ref=y_true, target=y_pred)


class TopIQNR(Loss):
    def __init__(
        self,
        name="TopIQ",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if backend.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        import pyiqa

        self.model = pyiqa.create_metric("topiq_nr", as_loss=True)

    def get_config(self):
        base_config = super().get_config()
        config = {"full_ref": False, "lower_better": self.model.lower_better}
        return {**base_config, **config}

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        del y_true
        # Library only supports channels first so change incoming data
        if backend.image_data_format() == "channels_last":
            y_pred = ops.moveaxis(y_pred, -1, 1)
        return 1 - self.model(target=y_pred)
