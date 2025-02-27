from auramask.metrics.pyiqa import IQAMetric


class TOPIQFR(IQAMetric):
    def __init__(self, name="TOPIQ_FR", **kwargs):
        super().__init__(name=name, metric_name="topiq_fr", **kwargs)


class TOPIQNR(IQAMetric):
    def __init__(self, name="TOPIQ_NR", **kwargs):
        super().__init__(name=name, metric_name="topiq_nr", **kwargs)
