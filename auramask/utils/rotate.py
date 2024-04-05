from keras_cv.layers import RandomRotation


class RandomRotatePairs(RandomRotation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def augment_segmentation_masks(self, segmentation_masks, transformations, **kwargs):
        return self._rotate_images(segmentation_masks, transformations)