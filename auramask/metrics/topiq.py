from keras import metrics, backend, ops

class TOPIQFR(metrics.MeanMetricWrapper):
    def __init__(
        self,
        name="TOPIQ_FR",
        **kwargs
    ):
        if backend.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        from pyiqa import create_metric

        self.model = create_metric("topiq_fr", as_loss=False)

        def topiq(y_true, y_pred):
            if backend.image_data_format() == "channels_last":
                y_true = ops.moveaxis(y_true, -1, 1)
                y_pred = ops.moveaxis(y_pred, -1, 1)
            diff = self.model(ref=y_true, target=y_pred)

        super().__init__(name=name, fn=topiq, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "full_ref": True,
            "lower_better": self.model.lower_better,
            "score_range":self.model.score_range,
        }
        return {**base_config, **config}

class TOPIQNR(metrics.MeanMetricWrapper):
    def __init__(
        self,
        name="TOPIQ_NR",
        **kwargs
    ):
        if backend.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        from pyiqa import create_metric

        self.model = create_metric("topiq_nr", as_loss=False)

        def topiq(y_true, y_pred):
            del y_true
            if backend.image_data_format() == "channels_last":
                y_pred = ops.moveaxis(y_pred, -1, 1)
            diff = self.model(target=y_pred)

        super().__init__(name=name, fn=topiq, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "full_ref": False,
            "lower_better": self.model.lower_better,
            "score_range":self.model.score_range,
        }
        return {**base_config, **config}