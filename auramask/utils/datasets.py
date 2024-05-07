from enum import Enum
import tensorflow as tf


class DatasetEnum(Enum):
    LFW = ("logasja/lfw", "default", ("image", "image"))
    INSTAGRAM = ("logasja/lfw", "aug", ("orig", "aug"))
