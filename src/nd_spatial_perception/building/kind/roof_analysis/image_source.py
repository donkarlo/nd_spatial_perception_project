from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio

from .georeferencing import GeoReferencer
from .models import GeoImage


class GeoTiffImageSource:
    """Loads one georeferenced RGB GeoTIFF from disk."""

    def load(self, image_path: Path) -> GeoImage:
        if not image_path.exists():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

        with rasterio.open(image_path) as dataset:
            if dataset.count < 3:
                raise ValueError("The GeoTIFF must contain at least three image bands.")
            rgb = dataset.read([1, 2, 3])
            rgb = np.moveaxis(rgb, 0, -1)
            rgb = self._to_uint8(rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gcps, gcp_crs = dataset.gcps
            tags = dataset.tags()
            quality = tags.get("GEOREFERENCE_QUALITY", "native")
            georeferencer = GeoReferencer.from_dataset(
                crs_value=dataset.crs,
                affine=dataset.transform,
                gcps=list(gcps),
                gcp_crs_value=gcp_crs,
                quality=quality,
            )
            description = tags.get(
                "SOURCE_DESCRIPTION",
                f"Georeferenced aerial GeoTIFF: {image_path.name}",
            )

        return GeoImage(
            bgr=bgr,
            source_path=str(image_path),
            source_description=description,
            georeferencer=georeferencer,
        )

    @staticmethod
    def _to_uint8(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array
        result = np.zeros(array.shape, dtype=np.uint8)
        for band in range(array.shape[2]):
            values = array[:, :, band].astype(np.float32)
            valid = values[np.isfinite(values)]
            if valid.size == 0:
                continue
            low, high = np.percentile(valid, [1, 99])
            if high <= low:
                continue
            scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
            result[:, :, band] = (scaled * 255.0).astype(np.uint8)
        return result
