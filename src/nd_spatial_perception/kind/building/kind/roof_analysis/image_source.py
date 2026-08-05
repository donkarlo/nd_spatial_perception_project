from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from .georeferencing import GeoReferencer
from .models import GeoImage


class GeoTiffValidationError(ValueError):
    """Raised when the selected input is not a usable RGB GeoTIFF."""


class GeoTiffImageSource:
    """Loads one georeferenced RGB GeoTIFF from disk."""

    @classmethod
    def validate(cls, image_path: Path) -> None:
        """Validate the file before roof analysis starts."""
        if not image_path.exists():
            raise GeoTiffValidationError(
                f"The input file does not exist:\n{image_path}"
            )

        if not image_path.is_file():
            raise GeoTiffValidationError(
                f"The input path is not a file:\n{image_path}"
            )

        if image_path.suffix.lower() not in {".tif", ".tiff"}:
            raise GeoTiffValidationError(
                "The selected file is not a GeoTIFF.\n"
                f"Received: {image_path}\n"
                "Expected: a .tif or .tiff file containing geographic "
                "coordinate information."
            )

        try:
            with rasterio.open(image_path) as dataset:
                cls._validate_dataset(dataset, image_path)
        except RasterioIOError as error:
            raise GeoTiffValidationError(
                "The selected file cannot be opened as a GeoTIFF.\n"
                f"Received: {image_path}\n"
                f"Rasterio error: {error}"
            ) from error

    def load(self, image_path: Path) -> GeoImage:
        self.validate(image_path)

        try:
            with rasterio.open(image_path) as dataset:
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
        except RasterioIOError as error:
            raise GeoTiffValidationError(
                f"Could not read the GeoTIFF:\n{image_path}\n{error}"
            ) from error

        return GeoImage(
            bgr=bgr,
            source_path=str(image_path),
            source_description=description,
            georeferencer=georeferencer,
        )

    @staticmethod
    def _validate_dataset(dataset: Any, image_path: Path) -> None:
        if dataset.driver != "GTiff":
            raise GeoTiffValidationError(
                "The file extension is .tif/.tiff, but the file is not in "
                "GeoTIFF format.\n"
                f"Received: {image_path}\n"
                f"Detected raster driver: {dataset.driver}"
            )

        if dataset.count < 3:
            raise GeoTiffValidationError(
                "The GeoTIFF must contain at least three image bands "
                "for RGB roof analysis.\n"
                f"Received bands: {dataset.count}"
            )

        gcps, gcp_crs = dataset.gcps
        has_usable_gcps = len(gcps) >= 4 and gcp_crs is not None
        has_usable_affine = (
                dataset.crs is not None and not dataset.transform.is_identity
        )

        if not has_usable_gcps and not has_usable_affine:
            raise GeoTiffValidationError(
                "The file is a TIFF image, but it is not georeferenced.\n"
                "A usable GeoTIFF must contain either:\n"
                "- a CRS and a non-identity affine transform, or\n"
                "- at least four ground-control points with a CRS.\n"
                f"Received: {image_path}"
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
