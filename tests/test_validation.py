from __future__ import annotations

from pathlib import Path

import pytest

from roof_analysis.image_source import GeoTiffImageSource, GeoTiffValidationError


def test_jpg_is_rejected_with_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "ordinary_image.jpg"
    path.write_bytes(b"not a geotiff")

    with pytest.raises(GeoTiffValidationError, match="not a GeoTIFF"):
        GeoTiffImageSource.validate(path)
