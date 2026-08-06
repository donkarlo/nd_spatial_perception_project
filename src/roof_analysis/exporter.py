from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from .models import BuildingCandidate, GeoImage, RoofResult


class ResultExporter:
    """Writes JSON results and visual QA overlays."""

    def export(
            self,
            geo_image: GeoImage,
            all_candidates: list[BuildingCandidate],
            selected_candidates: list[BuildingCandidate],
            results: list[RoofResult],
            output_dir: Path,
            footprint_source_path: Path,
    ) -> None:
        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

        result_document = {
            "input": {
                "geotiff": str(geo_image.path),
                "building_footprints": str(footprint_source_path),
            },
            "summary": {
                "usable_independent_buildings": len(all_candidates),
                "selected_roofs": len(results),
            },
            "buildings": [result.to_dict() for result in results],
        }
        (output_dir / "roof_attributes.json").write_text(
            json.dumps(result_document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        overview = geo_image.bgr.copy()
        for candidate in all_candidates:
            polygon = self._polygon_array(candidate.polygon_px)
            cv2.polylines(
                overview,
                [polygon],
                True,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        selected_by_id = {
            result.source_building_id: result
            for result in results
        }
        for candidate in selected_candidates:
            polygon = self._polygon_array(candidate.polygon_px)
            cv2.polylines(
                overview,
                [polygon],
                True,
                (0, 255, 0),
                4,
                cv2.LINE_AA,
            )
            result = selected_by_id[candidate.source_id]
            x, y = candidate.centroid_px
            cv2.putText(
                overview,
                result.building_id,
                (max(0, x - 45), max(24, y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(
            str(overlays_dir / "00_all_roofs_overlay.jpg"),
            overview,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )

        for index, (candidate, result) in enumerate(
                zip(selected_candidates, results, strict=True),
                start=1,
        ):
            self._write_crop(
                geo_image=geo_image,
                candidate=candidate,
                result=result,
                path=overlays_dir / f"{index:02d}_{result.building_id}.jpg",
            )

    def _write_crop(
            self,
            geo_image: GeoImage,
            candidate: BuildingCandidate,
            result: RoofResult,
            path: Path,
    ) -> None:
        polygon = self._polygon_array(candidate.polygon_px).reshape(-1, 2)
        x, y, width, height = cv2.boundingRect(polygon)
        padding = max(24, int(0.18 * max(width, height)))
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(geo_image.bgr.shape[1], x + width + padding)
        y1 = min(geo_image.bgr.shape[0], y + height + padding)

        crop = geo_image.bgr[y0:y1, x0:x1].copy()
        shifted = polygon - np.asarray([x0, y0], dtype=np.int32)
        cv2.polylines(
            crop,
            [shifted.reshape(-1, 1, 2)],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        label = (
            f"{result.building_id} | {result.roof_type} | "
            f"{result.material} | {result.area_m2:.1f} m2"
        )
        cv2.rectangle(crop, (0, 0), (min(crop.shape[1], 640), 38), (0, 0, 0), -1)
        cv2.putText(
            crop,
            label,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])

    @staticmethod
    def _polygon_array(polygon: Polygon) -> np.ndarray:
        return np.round(
            np.asarray(polygon.exterior.coords, dtype=np.float64)
        ).astype(np.int32).reshape(-1, 1, 2)
