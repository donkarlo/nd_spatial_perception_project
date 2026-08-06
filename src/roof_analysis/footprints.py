from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import requests
from pyproj import CRS, Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.ops import transform as shapely_transform

from .models import BuildingCandidate, FootprintCollection, GeoImage


class FootprintError(ValueError):
    """Raised when building footprints cannot support the requested analysis."""


class GeoJsonBuildingFootprintSource:
    """Loads, aligns and scores open building polygons from GeoJSON."""

    def __init__(
        self,
        min_area_m2: float = 20.0,
        min_area_px: float = 300.0,
        edge_margin_px: int = 12,
        independent_gap_m: float = 0.5,
    ) -> None:
        self.min_area_m2 = min_area_m2
        self.min_area_px = min_area_px
        self.edge_margin_px = edge_margin_px
        self.independent_gap_m = independent_gap_m

    def load(
        self,
        geo_image: GeoImage,
        footprints_path: Path,
    ) -> FootprintCollection:
        if not footprints_path.exists():
            raise FootprintError(
                f"The building-footprint file does not exist:\n{footprints_path}"
            )
        if not footprints_path.is_file():
            raise FootprintError(
                f"The building-footprint path is not a file:\n{footprints_path}"
            )
        if footprints_path.suffix.lower() not in {".geojson", ".json"}:
            raise FootprintError(
                "The building-footprint file must be GeoJSON (.geojson or .json)."
            )

        try:
            data = json.loads(footprints_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FootprintError(
                f"Could not read the building-footprint GeoJSON:\n{error}"
            ) from error

        source_crs = self._read_crs(data)
        source_description = str(
            data.get("name")
            or data.get("metadata", {}).get("source")
            or f"Building footprints: {footprints_path.name}"
        )
        candidates = self._build_candidates(
            geo_image=geo_image,
            features=self._features(data),
            source_crs=source_crs,
        )
        independent = self._keep_independent(candidates)
        independent.sort(key=lambda item: item.selection_score, reverse=True)

        return FootprintCollection(
            candidates=independent,
            source_description=source_description,
            source_path=footprints_path,
        )

    def materialize_mask(
        self,
        geo_image: GeoImage,
        candidate: BuildingCandidate,
    ) -> BuildingCandidate:
        """Create the full-resolution mask only for a selected building.

        Keeping one raster-sized mask for every source footprint can consume
        several gigabytes on a large orthophoto. Candidate ranking needs the
        mask only temporarily, so unselected candidates store an empty mask.
        """

        height, width = geo_image.bgr.shape[:2]
        mask = self._polygon_mask(
            polygon_px=candidate.polygon_px,
            width=width,
            height=height,
        )
        return replace(candidate, mask=mask)

    def _build_candidates(
        self,
        geo_image: GeoImage,
        features: list[dict[str, Any]],
        source_crs: CRS,
    ) -> list[BuildingCandidate]:
        height, width = geo_image.bgr.shape[:2]
        gray = cv2.cvtColor(geo_image.bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 140)
        image_polygon = geo_image.georeferencer.image_polygon_crs

        # A small inward buffer rejects footprints clipped by the raster boundary.
        pixel_size = max(
            abs(float(geo_image.georeferencer.affine.a)),
            abs(float(geo_image.georeferencer.affine.e)),
        )
        safe_image = image_polygon.buffer(-self.edge_margin_px * pixel_size)
        if safe_image.is_empty:
            safe_image = image_polygon

        target_crs = geo_image.georeferencer.crs
        transformer = None
        if source_crs != target_crs:
            transformer = Transformer.from_crs(
                source_crs,
                target_crs,
                always_xy=True,
            )

        candidates: list[BuildingCandidate] = []
        for feature_index, feature in enumerate(features, start=1):
            geometry_data = feature.get("geometry")
            if not geometry_data:
                continue
            try:
                geometry = shape(geometry_data)
            except Exception:
                continue
            if geometry.is_empty:
                continue
            if transformer is not None:
                geometry = shapely_transform(transformer.transform, geometry)

            for polygon_index, polygon in enumerate(
                self._polygon_parts(geometry),
                start=1,
            ):
                polygon = polygon if polygon.is_valid else polygon.buffer(0)
                if polygon.is_empty or not isinstance(polygon, Polygon):
                    continue
                if not safe_image.contains(polygon):
                    continue

                area_m2 = geo_image.georeferencer.area_m2(polygon)
                if area_m2 < self.min_area_m2:
                    continue

                polygon_px = self._polygon_to_pixel(
                    geo_image=geo_image,
                    polygon=polygon,
                )
                if polygon_px.is_empty:
                    continue
                area_px = float(polygon_px.area)
                if area_px < self.min_area_px:
                    continue

                mask = self._polygon_mask(
                    polygon_px=polygon_px,
                    width=width,
                    height=height,
                )
                if np.count_nonzero(mask) == 0:
                    continue

                boundary_support = self._boundary_support(
                    polygon_px=polygon_px,
                    edge_map=edges,
                    width=width,
                    height=height,
                )
                image_quality = self._image_quality(
                    image=geo_image.bgr,
                    gray=gray,
                    mask=mask,
                )
                area_score = float(
                    np.clip((area_m2 - self.min_area_m2) / 250.0, 0.0, 1.0)
                )
                compactness = self._compactness(polygon_px)
                selection_score = float(
                    np.clip(
                        0.38 * boundary_support
                        + 0.32 * image_quality
                        + 0.18 * area_score
                        + 0.12 * compactness,
                        0.0,
                        1.0,
                    )
                )

                properties = dict(feature.get("properties") or {})
                source_id = str(
                    properties.get("osm_id")
                    or properties.get("id")
                    or feature.get("id")
                    or f"feature_{feature_index}_{polygon_index}"
                )
                centroid = polygon_px.centroid
                candidates.append(
                    BuildingCandidate(
                        source_id=source_id,
                        polygon_crs=polygon,
                        polygon_px=polygon_px,
                        mask=np.empty((0, 0), dtype=np.uint8),
                        centroid_px=(
                            int(round(centroid.x)),
                            int(round(centroid.y)),
                        ),
                        source_properties=properties,
                        boundary_support=boundary_support,
                        image_quality=image_quality,
                        selection_score=selection_score,
                    )
                )
        return candidates

    def _keep_independent(
        self,
        candidates: list[BuildingCandidate],
    ) -> list[BuildingCandidate]:
        """Reject connected/overlapping building polygons.

        A building is considered independent when it does not overlap or touch
        another footprint and has at least the configured small gap. This is
        intentionally conservative and avoids selecting multiple parts of one
        building complex as separate roofs.
        """

        independent: list[BuildingCandidate] = []
        for index, candidate in enumerate(candidates):
            is_independent = True
            for other_index, other in enumerate(candidates):
                if index == other_index:
                    continue
                if candidate.polygon_crs.distance(other.polygon_crs) < self.independent_gap_m:
                    is_independent = False
                    break
            if is_independent:
                independent.append(candidate)
        return independent

    @staticmethod
    def _read_crs(data: dict[str, Any]) -> CRS:
        crs_data = data.get("crs")
        if not crs_data:
            return CRS.from_epsg(4326)
        properties = crs_data.get("properties") or {}
        name = properties.get("name")
        return CRS.from_user_input(name) if name else CRS.from_epsg(4326)

    @staticmethod
    def _features(data: dict[str, Any]) -> list[dict[str, Any]]:
        data_type = data.get("type")
        if data_type == "FeatureCollection":
            return list(data.get("features") or [])
        if data_type == "Feature":
            return [data]
        return [{"type": "Feature", "properties": {}, "geometry": data}]

    @staticmethod
    def _polygon_parts(geometry: Any) -> Iterable[Polygon]:
        if isinstance(geometry, Polygon):
            yield geometry
        elif isinstance(geometry, MultiPolygon):
            yield from geometry.geoms
        elif isinstance(geometry, GeometryCollection):
            for item in geometry.geoms:
                yield from GeoJsonBuildingFootprintSource._polygon_parts(item)

    @staticmethod
    def _polygon_to_pixel(geo_image: GeoImage, polygon: Polygon) -> Polygon:
        exterior = geo_image.georeferencer.crs_to_pixel(
            [(float(x), float(y)) for x, y in polygon.exterior.coords]
        )
        interiors = [
            geo_image.georeferencer.crs_to_pixel(
                [(float(x), float(y)) for x, y in ring.coords]
            )
            for ring in polygon.interiors
        ]
        result = Polygon(exterior, interiors)
        repaired = result if result.is_valid else result.buffer(0)
        if isinstance(repaired, Polygon):
            return repaired
        if isinstance(repaired, MultiPolygon) and repaired.geoms:
            return max(repaired.geoms, key=lambda item: item.area)
        return Polygon()

    @staticmethod
    def _ring_array(points: Iterable[tuple[float, float]]) -> np.ndarray:
        return np.round(np.asarray(list(points), dtype=np.float64)).astype(
            np.int32
        ).reshape(-1, 1, 2)

    def _polygon_mask(
        self,
        polygon_px: Polygon,
        width: int,
        height: int,
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self._ring_array(polygon_px.exterior.coords)], 255)
        for ring in polygon_px.interiors:
            cv2.fillPoly(mask, [self._ring_array(ring.coords)], 0)
        return mask

    def _boundary_support(
        self,
        polygon_px: Polygon,
        edge_map: np.ndarray,
        width: int,
        height: int,
    ) -> float:
        boundary = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(
            boundary,
            [self._ring_array(polygon_px.exterior.coords)],
            True,
            255,
            3,
        )
        dilated_edges = cv2.dilate(
            edge_map,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        boundary_pixels = boundary > 0
        count = int(np.count_nonzero(boundary_pixels))
        if count == 0:
            return 0.0
        return float(np.count_nonzero(dilated_edges[boundary_pixels]) / count)

    @staticmethod
    def _image_quality(
        image: np.ndarray,
        gray: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        pixels = image[mask > 0]
        gray_pixels = gray[mask > 0]
        if pixels.size == 0:
            return 0.0
        texture = float(np.std(gray_pixels))
        mean_brightness = float(np.mean(gray_pixels))
        clipped = float(
            np.mean((gray_pixels <= 8) | (gray_pixels >= 247))
        )
        texture_score = float(np.clip(texture / 45.0, 0.0, 1.0))
        brightness_score = float(
            np.clip(1.0 - abs(mean_brightness - 135.0) / 135.0, 0.0, 1.0)
        )
        clipping_score = float(np.clip(1.0 - 4.0 * clipped, 0.0, 1.0))
        return 0.45 * texture_score + 0.30 * brightness_score + 0.25 * clipping_score

    @staticmethod
    def _compactness(polygon_px: Polygon) -> float:
        perimeter = float(polygon_px.length)
        area = float(polygon_px.area)
        if perimeter <= 0.0 or area <= 0.0:
            return 0.0
        return float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0.0, 1.0))


class OverpassBuildingFootprintFetcher:
    """Downloads OSM building ways for exactly the GeoTIFF bounding box."""

    ENDPOINTS = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
    )

    def __init__(self, timeout_seconds: int = 120) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self, geo_image: GeoImage, output_path: Path) -> Path:
        min_lon, min_lat, max_lon, max_lat = geo_image.georeferencer.bounds_lonlat
        query = (
            "[out:json][timeout:90];"
            f"way[\"building\"]({min_lat},{min_lon},{max_lat},{max_lon});"
            "out tags geom;"
        )
        errors: list[str] = []
        payload: dict[str, Any] | None = None
        for endpoint in self.ENDPOINTS:
            try:
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    headers={
                        "User-Agent": (
                            "roof-analysis-propx-assessment/1.0 "
                            "(open building footprint download)"
                        )
                    },
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError) as error:
                errors.append(f"{endpoint}: {error}")
                time.sleep(1.0)

        if payload is None:
            raise FootprintError(
                "Could not download OpenStreetMap building footprints.\n"
                + "\n".join(errors)
            )

        features: list[dict[str, Any]] = []
        for element in payload.get("elements", []):
            geometry = element.get("geometry") or []
            coordinates = [
                [float(point["lon"]), float(point["lat"])]
                for point in geometry
                if "lon" in point and "lat" in point
            ]
            if len(coordinates) < 4:
                continue
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            properties = dict(element.get("tags") or {})
            properties["osm_id"] = f"way/{element.get('id')}"
            features.append(
                {
                    "type": "Feature",
                    "id": properties["osm_id"],
                    "properties": properties,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coordinates],
                    },
                }
            )

        if not features:
            raise FootprintError(
                "OpenStreetMap returned no building polygons for the image extent."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "type": "FeatureCollection",
            "name": "OpenStreetMap building footprints",
            "metadata": {
                "source": "OpenStreetMap contributors via Overpass API",
                "bbox_wgs84": [min_lon, min_lat, max_lon, max_lat],
            },
            "features": features,
        }
        output_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path
