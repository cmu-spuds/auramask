# ruff: noqa: F401

from auramask.losses.content import ContentLoss
from auramask.losses.perceptual import PerceptualLoss, IQAPerceptual
from auramask.losses.embeddistance import (
    FaceEmbeddingLoss,
    FaceEmbeddingThresholdLoss,
)
from auramask.losses.aesthetic import AestheticLoss, IQAAestheticLoss
from auramask.losses.ssim import DSSIMObjective, GRAYSSIMObjective, IQASSIMC, IQACWSSIM
from auramask.losses.style import StyleLoss, StyleRefs
from auramask.losses.variation import VariationLoss
from auramask.losses.zero_dce import (
    ColorConstancyLoss,
    SpatialConsistencyLoss,
    ExposureControlLoss,
    IlluminationSmoothnessLoss,
)
from auramask.losses.histogram import HistogramMatchingLoss
from auramask.losses.psnr import IQAPSNR
from auramask.losses.topiq import TopIQFR, TopIQNR
from auramask.losses.ms_swd import MSSWD
