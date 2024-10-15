from typing import List, cast

import numpy as np
import pylidc as pl
import scipy
from pylidc.utils import consensus
from scipy.ndimage import binary_fill_holes
from torch.utils.data import Dataset
from tqdm import tqdm

from .preprocessing import (
    preprocess_volume_and_mask,
    scan_labels_to_vector,
)
from .types import (
    BinaryMask,
    MalignancyInfo,
    NoduleCharacteristics,
    NoduleInfo,
    ScanLabelsVector,
    NormalizedVolume,
    ScanLabels,
    TransformPipeline,
    Volume,
)


class LIDCDataset(Dataset):
    """
    A PyTorch Dataset for the LIDC-IDRI lung cancer dataset.

    This dataset loads CT scans and their corresponding nodule annotations
    from the LIDC-IDRI dataset using the pylidc library. It provides access
    to the CT volumes, segmentation masks, and malignancy labels.

    Attributes:
        patient_ids (list): List of patient IDs to include in the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
        scans (list): List of pylidc Scan objects corresponding to patient_ids.
    """

    patient_ids: List[str]
    transform: TransformPipeline | None

    def __init__(self, patient_ids: List[str], transform: TransformPipeline | None = None):
        """
        Initialize the LIDCDataset.

        Args:
            patient_ids (list): List of patient IDs to include in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patient_ids = patient_ids
        self.transform = transform

    def __len__(self):
        """
        Return the number of scans in the dataset.

        Returns:
            int: Number of scans in the dataset.
        """
        return len(self.patient_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[Volume | NormalizedVolume, BinaryMask, ScanLabelsVector]:
        """
        Fetch a sample from the dataset.

        This method loads a CT volume, creates a segmentation mask,
        and generates classification labels for the nodules.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (volume, mask, labels) where volume is the CT scan,
                   mask is the segmentation mask, and labels are the
                   classification labels for the nodules.
        """
        # Get the patient ID
        patient_id = self.patient_ids[idx]

        # Query the Scan object within this method
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()

        if scan is None:
            raise ValueError(f"No scan found for patient ID {patient_id}")
        else:
            scan = cast(pl.Scan, scan)

        mask = self._create_mask(scan)
        labels = self._create_labels(scan)
        labels = scan_labels_to_vector(labels)

        # Get and preprocess the volume and preprocess the mask
        volume, mask = preprocess_volume_and_mask(scan, mask, (512, 512, 512))

        if self.transform:
            volume, mask, labels = self.transform(volume, mask, labels)

        # Add channel dimension to the volume (shape becomes [1, 512, 512, 512])
        volume = np.expand_dims(volume, axis=0)

        return (
            volume.astype(np.float32, copy=False),
            mask.astype(np.bool_, copy=False),
            labels.astype(np.float32, copy=False),
        )

    def _create_mask(self, scan: pl.Scan) -> BinaryMask:
        """
        Create a 3D segmentation mask from nodule annotations.

        This method creates a binary mask where voxels belonging to annotated nodules
        are set to 1, and all other voxels are 0. It uses a consensus approach,
        including a voxel if at least 50% of the radiologists marked it as part of a nodule.

        Args:
            scan (pylidc.Scan): A pylidc Scan object.

        Returns:
            np.ndarray: A 3D numpy array representing the segmentation mask.
        """
        volume_shape = scan.to_volume().shape
        # Get the shape of the full CT volume

        full_mask = np.zeros(volume_shape, dtype=np.bool)
        # Initialize an empty mask

        nodules = scan.cluster_annotations()
        # Group annotations by nodule

        for nodule in nodules:
            consensus_results = consensus(nodule, clevel=0.5, pad=[(0, 0), (0, 0), (0, 0)])
            # In the case that a nodule has conflicting annotations,
            # (radiologists disagree on the malignancy of the nodule),
            # we use the consensus function from pylidc.utils to compute
            # --well, the consensus between the annotations.
            #
            # - nodule: a list of Annotation objects
            # - clevel: This is the consensus level. It determines the
            #   threshold for including a voxel in the final mask:
            #      > 0.5 means that a voxel will be included in the consensus
            #      mask if at least 50% of the annotations marked it as part of the nodule.
            # - pad: This parameter specifies padding for the bounding box of the nodule:
            #      > It's a list of three tuples, one for each dimension (x, y, z).
            #      > Each tuple specifies the padding before and after the nodule
            #        in that dimension.
            #      > In this case, we're specifying no padding (0 before, 0 after)
            #        for all dimensions.
            #      > We might adjust this if we want to include some context
            #        around the nodule. For example, [(1,1), (1,1), (1,1)] would
            #        add one voxel of padding in all directions.

            if len(consensus_results) == 3:
                cmask, cbbox, _ = consensus_results
                # Unpack the results
            else:
                cmask, cbbox = consensus_results
                # Unpack the results

            full_mask[cbbox] = np.logical_or(full_mask[cbbox], cmask)
            # Place the nodule mask in the full mask

        filled_mask = binary_fill_holes(full_mask)
        # Fill any holes in the mask

        if filled_mask is None:
            # Handle the case where binary_fill_holes returns None
            return full_mask.astype(np.bool_)
            # When binary_fill_holes returns None, it typically means one of two things:
            #    a) The input array (full_mask) was empty (had a size of 0 in any dimension).
            #    b) The input array contained only False values (this would mean no nodules were detected).
        else:
            return filled_mask.astype(np.bool_)

    def _create_labels(self, scan: pl.Scan) -> ScanLabels:
        """
        Create classification labels for each nodule in the scan.

        This method assigns malignancy labels and other characteristics
        to each annotated nodule based on the radiologists' assessments.

        Args:
            scan (pylidc.Scan): A pylidc Scan object.

        Returns:
            ScanLabels: A Pydantic model containing a list of NoduleInfo objects,
                        each with detailed information about a nodule, including
                        aggregated assessments from multiple radiologists.
        """
        nodules = scan.cluster_annotations()
        # Group annotations by nodule

        labels = []
        for nodule_idx, nodule_anns in enumerate(nodules):
            malignancy_scores = [ann.malignancy for ann in nodule_anns]

            nodule_info = NoduleInfo(
                nodule_id=nodule_idx,
                bbox=nodule_anns[0].bbox(),  # Using first annotation's bbox
                centroid=np.mean([ann.centroid for ann in nodule_anns], axis=0),
                diameter_mm=np.mean([ann.diameter for ann in nodule_anns]),
                volume_mm3=np.mean([ann.volume for ann in nodule_anns]),
                is_small=all(ann.diameter < 3 for ann in nodule_anns),
                num_annotations=len(nodule_anns),
                malignancy=MalignancyInfo(
                    mean=np.mean(malignancy_scores),
                    median=np.median(malignancy_scores),
                    mode=scipy.stats.mode(malignancy_scores).mode.item(),
                    std=np.std(malignancy_scores),
                ),
                characteristics=NoduleCharacteristics(
                    subtlety=np.mean([ann.subtlety for ann in nodule_anns]),
                    internal_structure=np.mean([ann.internalStructure for ann in nodule_anns]),
                    calcification=np.mean([ann.calcification for ann in nodule_anns]),
                    sphericity=np.mean([ann.sphericity for ann in nodule_anns]),
                    margin=np.mean([ann.margin for ann in nodule_anns]),
                    lobulation=np.mean([ann.lobulation for ann in nodule_anns]),
                    spiculation=np.mean([ann.spiculation for ann in nodule_anns]),
                    texture=np.mean([ann.texture for ann in nodule_anns]),
                ),
            )
            labels.append(nodule_info)

        return ScanLabels(nodules=labels)
