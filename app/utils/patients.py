import pylidc as pl
from typing import Tuple, List
import random


def get_patient_ids(
    max_patients: int | None = None, train_ratio: float = 0.8, random_selection: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Get a list of patient IDs from the LIDC-IDRI dataset, split into training and validation sets.

    Args:
        max_patients (int, optional): Maximum number of patients to include. If None, include all patients.
        train_ratio (float): Ratio of patients to use for training. Default is 0.8 (80% for training).

    Returns:
        Tuple[List[str], List[str]]: Two lists of patient IDs, (train_ids, val_ids)

    Raises:
        ValueError: If train_ratio is not between 0 and 1.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    # Query all patient IDs from the dataset
    all_patient_ids = [scan.patient_id for scan in pl.query(pl.Scan).all()]

    # Remove duplicates (in case a patient has multiple scans)
    all_patient_ids = list(set(all_patient_ids))

    # Shuffle the list to ensure random selection
    if random_selection:
        random.shuffle(all_patient_ids)

    # Limit the number of patients if max_patients is specified
    if max_patients is not None:
        all_patient_ids = all_patient_ids[:max_patients]

    # Calculate the split point
    split_point = int(len(all_patient_ids) * train_ratio)

    # Split the IDs into train and validation sets
    train_ids = all_patient_ids[:split_point]
    val_ids = all_patient_ids[split_point:]

    return train_ids, val_ids
