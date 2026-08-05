from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pyproj import CRS, Transformer
from rasterio.control import GroundControlPoint
from rasterio.transform import Affine
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform


class GeoreferencingError(ValueError):
    """Raised when an image cannot provide geographic coordinates."""


@dataclass(frozen=True, slots=True)
class GeoReferencer:
    """Converts pixel polygons to lon/lat and computes planimetric area."""

    crs: CRS
    affine: Affine | None = None
    homography_to_crs: np.ndarray | None = None
    quality: str = "native"

    @classmethod
    def from_dataset(
        cls,
        crs_value: object,
        affine: Affine,
        gcps: list[GroundControlPoint],
        gcp_crs_value: object | None,
        quality: str,
    ) -> "GeoReferencer":
        if gcps and gcp_crs_value is not None:
            if len(gcps) < 4:
                raise GeoreferencingError(
                    "A GeoTIFF using ground-control points needs at least four GCPs."
                )
            pixels = np.float32([[gcp.col, gcp.row] for gcp in gcps])
            world = np.float32([[gcp.x, gcp.y] for gcp in gcps])
            homography, _ = cv2.findHomography(pixels, world, method=0)
            if homography is None:
                raise GeoreferencingError("Could not solve GeoTIFF ground-control points.")
            return cls(
                crs=CRS.from_user_input(gcp_crs_value),
                homography_to_crs=homography,
                quality=quality,
            )

        if crs_value is None or affine.is_identity:
            raise GeoreferencingError(
                "The input must be a georeferenced GeoTIFF with a CRS and "
                "an affine transform or ground-control points."
            )
        return cls(
            crs=CRS.from_user_input(crs_value),
            affine=affine,
            quality=quality,
        )

    def pixel_to_crs(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if self.homography_to_crs is not None:
            source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(
                source, self.homography_to_crs
            ).reshape(-1, 2)
            return [(float(x), float(y)) for x, y in transformed]

        if self.affine is None:
            raise GeoreferencingError("No usable georeferencing transform is available.")
        return [
            tuple(map(float, self.affine * (x, y)))
            for x, y in points
        ]

    def pixel_to_lonlat(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        world_points = self.pixel_to_crs(points)
        if self.crs.to_epsg() == 4326:
            return world_points
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return [transformer.transform(x, y) for x, y in world_points]

    def area_m2(self, points: list[tuple[float, float]]) -> float:
        world_points = self.pixel_to_crs(points)
        polygon = Polygon(world_points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return 0.0

        if self.crs.is_projected:
            units = [axis.unit_name.lower() for axis in self.crs.axis_info]
            if units and all("metre" in unit or "meter" in unit for unit in units):
                return abs(float(polygon.area))

        lonlat = self.pixel_to_lonlat(points)
        lonlat_polygon = Polygon(lonlat)
        centroid = lonlat_polygon.centroid
        utm_zone = int((centroid.x + 180.0) // 6.0) + 1
        epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        metric_polygon = shapely_transform(transformer.transform, lonlat_polygon)
        return abs(float(metric_polygon.area))

    @property
    def area_confidence_factor(self) -> float:
        return 0.62 if self.quality.lower() == "approximate" else 0.95
