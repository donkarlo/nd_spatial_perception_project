from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

PixelPoint = tuple[float, float]
LonLatPoint = tuple[float, float]


@dataclass(frozen=True, slots=True)
class GeoImage:
    """A BGR image together with its georeferencing metadata."""

    bgr: np.ndarray
    path: Path
    description: str
    georeferencer: Any


@dataclass(frozen=True, slots=True)
class BuildingCandidate:
    """One independent building footprint aligned to the input image."""

    source_id: str
    polygon_crs: Polygon
    polygon_px: Polygon
    mask: np.ndarray
    centroid_px: tuple[int, int]
    source_properties: dict[str, Any]
    boundary_support: float
    image_quality: float
    selection_score: float


@dataclass(frozen=True, slots=True)
class FootprintCollection:
    """All usable independent building footprints inside one image."""

    candidates: list[BuildingCandidate]
    source_description: str
    source_path: Path


@dataclass(frozen=True, slots=True)
class RoofResult:
    """Structured roof result for one selected building."""

    building_id: str
    source_building_id: str
    source_used: list[str]
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
            "source_building_id": self.source_building_id,
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
                "slope_deg": self.slope_deg,
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
                key: round(float(value), 3)
                for key, value in self.confidence.items()
            },
            "notes": self.notes,
        }
