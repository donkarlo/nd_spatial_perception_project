from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from .georeferencing import GeoReferencer, GeoreferencingError
from .models import GeoImage


class GeoTiffValidationError(ValueError):
    """Raised when the input is not a usable georeferenced RGB GeoTIFF."""


class GeoTiffImageSource:
    """Loads and validates a georeferenced RGB GeoTIFF."""

    @classmethod
    def validate(cls, image_path: Path) -> None:
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
                "The selected image is not a GeoTIFF.\n"
                f"Received: {image_path}\n"
                "Expected: a .tif or .tiff file with a CRS and map transform."
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
                georeferencer = GeoReferencer.from_dataset(
                    crs_value=dataset.crs,
                    affine=dataset.transform,
                    width=dataset.width,
                    height=dataset.height,
                )
                tags = dataset.tags()
                description = tags.get(
                    "SOURCE_DESCRIPTION",
                    f"Georeferenced RGB GeoTIFF: {image_path.name}",
                )
        except (RasterioIOError, GeoreferencingError) as error:
            raise GeoTiffValidationError(str(error)) from error

        return GeoImage(
            bgr=bgr,
            path=image_path,
            description=description,
            georeferencer=georeferencer,
        )

    @staticmethod
    def _validate_dataset(dataset: Any, image_path: Path) -> None:
        if dataset.driver != "GTiff":
            raise GeoTiffValidationError(
                "The file has a TIFF extension but is not a GeoTIFF dataset.\n"
                f"Received: {image_path}\n"
                f"Detected driver: {dataset.driver}"
            )
        if dataset.count < 3:
            raise GeoTiffValidationError(
                "The GeoTIFF must contain at least three bands for RGB analysis.\n"
                f"Received bands: {dataset.count}"
            )
        if dataset.crs is None:
            raise GeoTiffValidationError(
                "The TIFF has no CRS and therefore cannot be aligned with "
                "building footprints."
            )
        if dataset.transform.is_identity:
            raise GeoTiffValidationError(
                "The TIFF has no usable geographic transform."
            )
        if dataset.width < 128 or dataset.height < 128:
            raise GeoTiffValidationError(
                "The GeoTIFF is too small for roof analysis. "
                "Both dimensions must be at least 128 pixels."
            )

    @staticmethod
    def _to_uint8(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array

        result = np.zeros(array.shape, dtype=np.uint8)
        for band_index in range(array.shape[2]):
            values = array[:, :, band_index].astype(np.float32)
            valid = values[np.isfinite(values)]
            if valid.size == 0:
                continue
            low, high = np.percentile(valid, [1.0, 99.0])
            if high <= low:
                continue
            scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
            result[:, :, band_index] = np.round(scaled * 255.0).astype(np.uint8)
        return result
