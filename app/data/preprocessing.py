from typing import TypeVar

import numpy as np
import pylidc as pl
from scipy.ndimage import zoom
from tqdm import tqdm

from .types import (
    MalignancyInfo,
    NDArray3,
    NoduleCharacteristics,
    NoduleInfo,
    ScanLabelsVector,
    NormalizedVolume,
    ScanLabels,
    Volume,
    Mask,
)

NVolumeOrMask = TypeVar("NVolumeOrMask", NormalizedVolume, Mask)


def crop_or_pad_volume_to_size(
    volume: NVolumeOrMask, target_shape: tuple[int, int, int]
) -> NVolumeOrMask:
    """
    Crop or pad the volume to the target shape.

    Args:
        volume (np.ndarray): The input volume.
        target_shape (tuple[int, int, int]): The desired output shape.

    Returns:
        np.ndarray: The volume cropped or padded to the target shape.
    """
    # Initialize new volume with zeros
    new_volume = np.zeros(target_shape, dtype=volume.dtype)

    # Calculate the center indices for cropping or padding
    src_slices = []
    dst_slices = []
    for i in range(3):
        src_size = volume.shape[i]
        dst_size = target_shape[i]

        if src_size >= dst_size:
            # Crop the volume
            start_idx = (src_size - dst_size) // 2
            src_slice = slice(start_idx, start_idx + dst_size)
            dst_slice = slice(0, dst_size)
        else:
            # Pad the volume
            start_idx = (dst_size - src_size) // 2
            src_slice = slice(0, src_size)
            dst_slice = slice(start_idx, start_idx + src_size)

        src_slices.append(src_slice)
        dst_slices.append(dst_slice)

    # Place the volume in the center of new_volume
    new_volume[tuple(dst_slices)] = volume[tuple(src_slices)]

    return new_volume  # type: ignore


def resample_volume_and_mask(
    scan: pl.Scan,
    mask: Mask,
    scan_spacings: NDArray3[np.float32],
    new_spacings: NDArray3[np.float32] = np.array([1.0, 1.0, 1.0]),
) -> tuple[Volume, Mask]:
    """
    Resample the volume and mask to have new spacings, ensuring integer output dimensions.

    Args:
        volume (np.ndarray): The input volume.
        mask (np.ndarray): The input mask.
        scan_spacings (np.ndarray): The spacings of the scan.
        new_spacings (np.ndarray): The desired voxel spacing.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The resampled volume and mask.
    """
    # Convert the scan to a numpy array
    volume = scan.to_volume()

    # Calculate initial resize factors
    resize_factor = scan_spacings / new_spacings

    # Calculate the new real shape
    new_real_shape = volume.shape * resize_factor

    # Round the new shape to the nearest integer
    new_shape = np.round(new_real_shape)

    # Adjust the resize factors to match the integer new shape
    adjusted_resize_factor = new_shape / volume.shape

    # Resample volume (order=1 for linear interpolation)
    zoomed_volume = zoom(volume, adjusted_resize_factor, order=1, mode="nearest")

    # Resample mask (order=0 for nearest-neighbor interpolation)
    zoomed_mask = zoom(mask, adjusted_resize_factor, order=0, mode="nearest")

    return np.round(zoomed_volume).astype(np.int16), np.round(zoomed_mask).astype(np.bool_)


def normalize_hu(volume: Volume) -> NormalizedVolume:
    """
    Normalize the HU values of the CT volume.
    CT scans use Hounsfield Units (HU) to represent radiodensity.
    These units typically range from -1000 (air) to +3000 (dense bone),
    but most soft tissues fall within a smaller range. Normalizing
    these values to a standard range helps the machine learning model
    focus on the relevant intensity variations and improves training stability.

    Args:
        volume (np.ndarray): The input CT volume in Hounsfield Units.

    Returns:
        np.ndarray: The normalized volume.
    """
    # Define the HU range we're interested in
    # -1000 HU is typically air, 400 HU covers most soft tissues
    min_hu = -1000
    max_hu = 400

    # Clip the volume so all values are between min_hu and max_hu
    volume = np.clip(volume, min_hu, max_hu)

    # Normalize the values to be between 0 and 1
    normalized_volume = (volume - min_hu) / (max_hu - min_hu)

    return normalized_volume


def preprocess_volume_and_mask(
    scan: pl.Scan,
    mask: Mask,
    target_shape: tuple[int, int, int],
    new_spacings: NDArray3[np.float32] = np.array([1.0, 1.0, 1.0]),
) -> tuple[NormalizedVolume, Mask]:
    """
    Preprocess the volume and mask: resample, normalize, and crop/pad.
    """
    scan_spacings = scan.spacings

    # Resample volume and mask with adjusted resize factors
    volume, mask = resample_volume_and_mask(scan, mask, scan_spacings, new_spacings)

    # Normalize the volume
    volume = normalize_hu(volume)

    # Crop or pad volume and mask to target shape
    volume = crop_or_pad_volume_to_size(volume, target_shape)
    mask = crop_or_pad_volume_to_size(mask, target_shape)

    return volume, mask


def scan_labels_to_vector(scan_labels: ScanLabels, max_nodules: int = 5) -> ScanLabelsVector:
    """
    Convert the ScanLabels object into a fixed-size vector representation.

    Args:
        scan_labels (ScanLabels): The labels from the scan with multiple nodules.
        max_nodules (int): Maximum number of nodules to represent in the vector,
                           padding will be applied if fewer nodules exist.

    Returns:
        np.ndarray: A vector representation of the nodules, padded if needed.
    """

    def flatten_nodule(nodule: NoduleInfo) -> np.ndarray:
        """
        Flatten the NoduleInfo object into a vector.
        Extract malignancy, characteristics, and basic nodule info.
        """
        malignancy_info = [
            nodule.malignancy.mean,
            nodule.malignancy.median,
            nodule.malignancy.mode,
            nodule.malignancy.std,
        ]

        characteristics_info = [
            nodule.characteristics.subtlety,
            nodule.characteristics.internal_structure,
            nodule.characteristics.calcification,
            nodule.characteristics.sphericity,
            nodule.characteristics.margin,
            nodule.characteristics.lobulation,
            nodule.characteristics.spiculation,
            nodule.characteristics.texture,
        ]

        basic_info = [
            nodule.diameter_mm,
            nodule.volume_mm3,
            nodule.is_small,
            nodule.num_annotations,
        ]

        return np.array(malignancy_info + characteristics_info + basic_info)

    nodule_vectors = []

    # Convert each nodule into a vector and collect them
    for nodule in scan_labels.nodules:
        nodule_vector = flatten_nodule(nodule)
        nodule_vectors.append(nodule_vector)

    # Define a default nodule vector in case there are no nodules
    if len(nodule_vectors) > 0:
        default_nodule_vector = np.zeros_like(nodule_vectors[0])
    else:
        # Assuming a default size for empty nodule vectors
        default_nodule_vector = np.zeros(
            4 + 8 + 4
        )  # Malignancy (4), Characteristics (8), Basic Info (4)

    # Handle padding for cases where there are fewer nodules than max_nodules
    if len(nodule_vectors) < max_nodules:
        padding_needed = max_nodules - len(nodule_vectors)
        nodule_vectors.extend([default_nodule_vector] * padding_needed)

    # If there are more nodules than max_nodules, we truncate the list
    nodule_vectors = nodule_vectors[:max_nodules]

    # Concatenate all nodule vectors into a single vector
    return np.concatenate(nodule_vectors)
