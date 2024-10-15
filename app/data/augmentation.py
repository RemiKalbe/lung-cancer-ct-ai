from typing import List

import numpy as np
from scipy.ndimage import rotate, zoom

from app.data.preprocessing import crop_or_pad_volume_to_size

from .types import (
    BinaryMask,
    ScanLabelsVector,
    NormalizedVolume,
    TransformPipeline,
    TransformPipelineOutput,
    Volume,
)


class Compose:
    """
    Composes several transforms together.

    This class allows you to chain multiple augmentation techniques and apply them in sequence.

    Args:
        transforms (List[callable]): List of transforms to compose.
    """

    def __init__(self, transforms: List[TransformPipeline]):
        self.transforms = transforms

    def __call__(
        self, volume: Volume | NormalizedVolume, mask: BinaryMask, labels: ScanLabelsVector
    ) -> TransformPipelineOutput:
        for t in self.transforms:
            volume, mask, labels = t(volume, mask, labels)
        return volume, mask, labels


class RandomRotation3D:
    """
    Randomly rotate the volume and mask.

    This augmentation helps the model become invariant to the orientation of the scan.

    Args:
        max_angle (float): Maximum rotation angle in degrees. Default is 20.
    """

    def __init__(self, max_angle: float = 20):
        self.max_angle = max_angle

    def __call__(
        self, volume: Volume | NormalizedVolume, mask: BinaryMask, labels: ScanLabelsVector
    ) -> TransformPipelineOutput:
        # Choose a random angle
        angle = np.random.uniform(-self.max_angle, self.max_angle)

        # Randomly choose 2 axes to rotate around
        axes = np.random.choice([0, 1, 2], size=2, replace=False)

        # Rotate volume (order=1 for linear interpolation)
        volume = rotate(
            volume, angle, axes=tuple(axes), reshape=False, order=1, mode="nearest"
        ).astype(np.float32, copy=False)

        # Rotate mask (order=0 for nearest neighbor interpolation to preserve label values)
        mask = rotate(mask, angle, axes=tuple(axes), reshape=False, order=0, mode="nearest").astype(
            np.bool_, copy=False
        )

        return volume, mask, labels


class RandomFlip3D:
    """
    Randomly flip the volume and mask along one of the axes.

    This augmentation helps the model become invariant to the orientation of the scan.
    It's useful for introducing variety in the positioning of organs and structures.
    """

    def __call__(
        self, volume: Volume | NormalizedVolume, mask: BinaryMask, labels: ScanLabelsVector
    ) -> TransformPipelineOutput:
        # Randomly choose an axis to flip
        axis = np.random.choice([0, 1, 2])

        # Flip both volume and mask along the chosen axis
        if volume.dtype == np.int16:
            volume = np.flip(volume, axis=axis).astype(np.int16)
        else:
            volume = np.flip(volume, axis=axis).astype(np.float32)

        mask = np.flip(mask, axis=axis)

        return volume, mask, labels


class RandomZoom3D:
    """
    Randomly zoom in or out on the volume and mask.

    This augmentation helps the model become robust to variations in the size of structures.
    It's particularly useful for dealing with different scanner resolutions or patient sizes.

    Args:
        min_factor (float): Minimum zoom factor. Default is 0.9 (10% zoom out).
        max_factor (float): Maximum zoom factor. Default is 1.1 (10% zoom in).
    """

    def __init__(self, min_factor: float = 0.9, max_factor: float = 1.1):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(
        self, volume: Volume | NormalizedVolume, mask: BinaryMask, labels: ScanLabelsVector
    ) -> TransformPipelineOutput:
        # Choose random zoom factors for each dimension
        factor = np.random.uniform(self.min_factor, self.max_factor, 3)

        # Apply zoom to volume (order=1 for linear interpolation)
        zoomed_volume = zoom(volume, factor, order=1, mode="nearest").astype(np.float32, copy=False)

        # Apply zoom to mask (order=0 for nearest neighbor interpolation to preserve label values)
        zoomed_mask = zoom(mask, factor, order=0, mode="nearest").astype(np.bool_, copy=False)

        # Crop or pad the zoomed volume and mask to the original shape
        original_shape = volume.shape
        volume = crop_or_pad_volume_to_size(zoomed_volume, original_shape)
        mask = crop_or_pad_volume_to_size(zoomed_mask, original_shape)

        return volume, mask, labels


class RandomNoise:
    """
    Add random Gaussian noise to the volume.

    This augmentation helps the model become robust to noise in the scans,
    which can vary depending on the scanning equipment and settings.

    Args:
        noise_variance (float): Variance of the Gaussian noise. Default is 0.01.
    """

    def __init__(self, noise_variance: float = 0.01):
        self.noise_variance = noise_variance

    def __call__(
        self, volume: Volume | NormalizedVolume, mask: BinaryMask, labels: ScanLabelsVector
    ) -> TransformPipelineOutput:
        # Generate Gaussian noise
        noise = np.random.normal(0, self.noise_variance**0.5, volume.shape)
        # Add noise to the volume
        volume = volume + noise
        return volume, mask, labels


class RandomIntensityShift:
    """
    Randomly shift the intensity values in the volume.

    This augmentation helps the model become robust to variations in intensity levels,
    which can occur due to differences in scanning equipment or settings.

    Args:
        max_shift (float): Maximum intensity shift as a fraction of the full intensity range.
                           Default is 0.1 (10% of the intensity range).
    """

    def __init__(self, max_shift: float = 0.1):
        self.max_shift = max_shift

    def __call__(
        self, volume: Volume | NormalizedVolume, mask: BinaryMask, labels: ScanLabelsVector
    ) -> TransformPipelineOutput:
        # Choose a random shift value
        shift = np.random.uniform(-self.max_shift, self.max_shift)
        # Apply the shift to the volume
        volume = volume + shift
        # Clip values to ensure they stay in the [0, 1] range
        volume = np.clip(volume, 0, 1)
        return volume, mask, labels
