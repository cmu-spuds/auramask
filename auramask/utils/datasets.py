from enum import Enum


class DatasetEnum(Enum):
    LFW = ("logasja/lfw", "default", ("image", "image"))
    INSTAGRAM = ("logasja/lfw", "aug", ("orig", "aug"))
