from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform


class GeoreferencingError(ValueError):
    """Raised when a raster does not provide usable map coordinates."""


@dataclass(frozen=True, slots=True)
class GeoReferencer:
    """Converts between raster pixels, the raster CRS and WGS84."""

    crs: CRS
    affine: Affine
    width: int
    height: int

    @classmethod
    def from_dataset(
        cls,
        crs_value: object,
        affine: Affine,
        width: int,
        height: int,
    ) -> "GeoReferencer":
        if crs_value is None:
            raise GeoreferencingError("The GeoTIFF does not contain a CRS.")
        if affine.is_identity:
            raise GeoreferencingError(
                "The GeoTIFF has an identity transform and is not georeferenced."
            )
        return cls(
            crs=CRS.from_user_input(crs_value),
            affine=affine,
            width=width,
            height=height,
        )

    def pixel_to_crs(
        self,
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        return [tuple(map(float, self.affine * point)) for point in points]

    def crs_to_pixel(
        self,
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        inverse = ~self.affine
        return [tuple(map(float, inverse * point)) for point in points]

    def pixel_to_lonlat(
        self,
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        world = self.pixel_to_crs(points)
        if self.crs.to_epsg() == 4326:
            return world
        transformer = Transformer.from_crs(
            self.crs,
            "EPSG:4326",
            always_xy=True,
        )
        return [transformer.transform(x, y) for x, y in world]

    def lonlat_to_crs_geometry(self, geometry: Polygon) -> Polygon:
        if self.crs.to_epsg() == 4326:
            return geometry
        transformer = Transformer.from_crs(
            "EPSG:4326",
            self.crs,
            always_xy=True,
        )
        transformed = shapely_transform(transformer.transform, geometry)
        if not isinstance(transformed, Polygon):
            raise GeoreferencingError("Expected a polygon after reprojection.")
        return transformed

    def crs_to_lonlat_geometry(self, geometry: Polygon) -> Polygon:
        if self.crs.to_epsg() == 4326:
            return geometry
        transformer = Transformer.from_crs(
            self.crs,
            "EPSG:4326",
            always_xy=True,
        )
        transformed = shapely_transform(transformer.transform, geometry)
        if not isinstance(transformed, Polygon):
            raise GeoreferencingError("Expected a polygon after reprojection.")
        return transformed

    @property
    def image_polygon_crs(self) -> Polygon:
        corners = self.pixel_to_crs(
            [
                (0.0, 0.0),
                (float(self.width), 0.0),
                (float(self.width), float(self.height)),
                (0.0, float(self.height)),
                (0.0, 0.0),
            ]
        )
        return Polygon(corners)

    @property
    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        polygon = self.crs_to_lonlat_geometry(self.image_polygon_crs)
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        return min_lon, min_lat, max_lon, max_lat

    def area_m2(self, polygon_crs: Polygon) -> float:
        polygon = polygon_crs if polygon_crs.is_valid else polygon_crs.buffer(0)
        if polygon.is_empty:
            return 0.0

        if self.crs.is_projected:
            units = [axis.unit_name.lower() for axis in self.crs.axis_info]
            if units and all(
                "metre" in unit or "meter" in unit
                for unit in units
            ):
                return abs(float(polygon.area))

        lonlat = self.crs_to_lonlat_geometry(polygon)
        centroid = lonlat.centroid
        utm_zone = int((centroid.x + 180.0) // 6.0) + 1
        epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{epsg}",
            always_xy=True,
        )
        metric = shapely_transform(transformer.transform, lonlat)
        return abs(float(metric.area))
