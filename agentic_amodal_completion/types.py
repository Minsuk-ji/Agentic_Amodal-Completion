from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


BBox = List[int]


@dataclass
class ObjectItem:
    id: str
    name: str
    bbox: BBox
    mask_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundaryExpansion:
    need_expand: bool = False
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    top_ratio: float = 0.0
    bottom_ratio: float = 0.0
    left_ratio: float = 0.0
    right_ratio: float = 0.0
    extent: Dict[str, str] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "need_expand": self.need_expand,
            "expand_ratio": {
                "top": self.top_ratio,
                "bottom": self.bottom_ratio,
                "left": self.left_ratio,
                "right": self.right_ratio,
            },
            "extent": self.extent,
            "reason": self.reason,
        }

    def to_runtime_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundaryVerification:
    boundary_complete: bool
    expand_pixels: Dict[str, int] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    input_image: str
    instruction: str
    save_dir: str
    occluder_plan_path: str
    target_clean_path: str
    occluder_mask_path: str
    inpaint_mask_path: str
    description_path: str
    completed_path: Optional[str]
    debug: Dict[str, Any]
    completed_target_rgba_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
