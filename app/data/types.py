from typing import Callable, List, Tuple, TypedDict, TypeVar, Union
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

#
# Utils
#

DType = TypeVar("DType", bound=np.generic)
NDArray3 = npt.NDArray[DType]
Mask = npt.NDArray[np.bool_]

#
# Labels
#


@dataclass
class MalignancyInfo:
    mean: np.float32
    median: np.float32
    mode: np.float32
    std: np.float32


@dataclass
class NoduleCharacteristics:
    subtlety: np.float32
    internal_structure: np.float32
    calcification: np.float32
    sphericity: np.float32
    margin: np.float32
    lobulation: np.float32
    spiculation: np.float32
    texture: np.float32


@dataclass
class NoduleInfo:
    nodule_id: int
    bbox: Tuple[slice, slice, slice]
    diameter_mm: np.float32
    volume_mm3: np.float32
    is_small: bool
    num_annotations: int
    malignancy: MalignancyInfo
    characteristics: NoduleCharacteristics
    centroid: np.ndarray = field(default_factory=lambda: np.array([100.0, 100.0, 50.0]))


@dataclass
class ScanLabels:
    nodules: List[NoduleInfo]


#
# More general types
#

BinaryMask = npt.NDArray[np.bool_]
Volume = npt.NDArray[np.int16]
NormalizedVolume = npt.NDArray[np.float32]
ScanLabelsVector = npt.NDArray[np.float32]


class TransformPipelineInput(TypedDict):
    volume: Union[Volume, NormalizedVolume]
    mask: BinaryMask
    labels: ScanLabels


TransformPipelineOutput = Tuple[Union[Volume, NormalizedVolume], BinaryMask, ScanLabelsVector]
TransformPipeline = Callable[
    [Union[Volume, NormalizedVolume], BinaryMask, ScanLabelsVector],
    TransformPipelineOutput,
]
