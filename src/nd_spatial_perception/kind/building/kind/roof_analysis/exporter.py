from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .models import GeoImage, RoofResult


class ResultExporter:
    """Writes the requested JSON and 3–5 roof overlay images."""

    def export(
        self,
        geo_image: GeoImage,
        results: list[RoofResult],
        output_dir: Path,
    ) -> None:
        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "roof_attributes.json"
        json_path.write_text(
            json.dumps(
                [result.to_dict() for result in results],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        overview = geo_image.bgr.copy()
        for result in results:
            polygon = self._polygon_array(result)
            cv2.polylines(overview, [polygon], True, (0, 255, 0), 3)
            x, y = map(int, result.polygon_px[0])
            cv2.putText(
                overview,
                result.building_id,
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(overlays_dir / "00_all_roofs_overlay.jpg"), overview)

        for number, result in enumerate(results[:4], start=1):
            polygon = self._polygon_array(result).reshape(-1, 2)
            x, y, width, height = cv2.boundingRect(polygon)
            padding = 30
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
            )
            cv2.putText(
                crop,
                result.building_id,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(
                str(overlays_dir / f"{number:02d}_{result.building_id}.jpg"),
                crop,
            )

    @staticmethod
    def _polygon_array(result: RoofResult) -> np.ndarray:
        return np.asarray(result.polygon_px, dtype=np.int32).reshape(-1, 1, 2)
