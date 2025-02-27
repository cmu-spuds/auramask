from keras import metrics, backend, ops


def iqa_metric(y_true, y_pred, metric_obj):
    y_true = ops.convert_to_tensor(y_true, "float32")
    y_pred = ops.convert_to_tensor(y_pred, "float32")
    if backend.image_data_format() == "channels_last":
        y_true = ops.moveaxis(y_true, -1, 1)
        y_pred = ops.moveaxis(y_pred, -1, 1)
    return metric_obj(ref=y_true, target=y_pred)


class IQAMetric(metrics.MeanMetricWrapper):
    def __init__(self, metric_name, name="IQA", **kwargs):
        if backend.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        from pyiqa import create_metric

        inner_metric = create_metric(metric_name, as_loss=False)

        super().__init__(name=name, fn=iqa_metric, metric_obj=inner_metric, **kwargs)

    def get_config(self):
        config = {
            "full_ref": True,
            "lower_better": self._fn_kwargs["metric_obj"].lower_better,
            "score_range": self._fn_kwargs["metric_obj"].score_range,
        }
        return config
