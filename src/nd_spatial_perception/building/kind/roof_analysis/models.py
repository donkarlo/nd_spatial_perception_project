from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PixelPoint = tuple[float, float]
LonLatPoint = tuple[float, float]


@dataclass(frozen=True, slots=True)
class GeoImage:
    """RGB image plus the georeferencing needed for lon/lat and metric area."""

    bgr: np.ndarray
    source_path: str
    source_description: str
    georeferencer: Any


@dataclass(frozen=True, slots=True)
class RoofCandidate:
    """One image-space roof candidate produced by the detector."""

    contour: np.ndarray
    mask: np.ndarray
    detection_score: float
    detector_label: str
    centroid_px: tuple[int, int]


@dataclass(frozen=True, slots=True)
class RoofResult:
    """Structured result for one detected building roof."""

    building_id: str
    source_used: str
    polygon_lonlat: list[LonLatPoint]
    polygon_px: list[PixelPoint]
    area_m2: float
    roof_type: str
    material: str
    orientation: str
    orientation_deg: float | None
    slope_deg: float | None
    solar_panels_present: bool
    solar_panel_count: int
    superstructure_count: int
    visible_condition: str
    confidence: dict[str, float]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "source_used": self.source_used,
            "roof": {
                "polygon": [
                    [round(lon, 7), round(lat, 7)]
                    for lon, lat in self.polygon_lonlat
                ],
                "area_m2": round(self.area_m2, 2),
                "type": self.roof_type,
                "material": self.material,
                "orientation": self.orientation,
                "orientation_deg": (
                    round(self.orientation_deg, 1)
                    if self.orientation_deg is not None
                    else None
                ),
                "slope_deg": (
                    round(self.slope_deg, 1)
                    if self.slope_deg is not None
                    else None
                ),
                "solar_panels": {
                    "present": self.solar_panels_present,
                    "count": self.solar_panel_count,
                },
                "superstructures": {
                    "count": self.superstructure_count,
                },
                "visible_condition": self.visible_condition,
            },
            "confidence": {
                key: round(value, 3) for key, value in self.confidence.items()
            },
            "notes": self.notes,
        }
