from __future__ import annotations

from pathlib import Path

from .attributes import RoofAttributeExtractor
from .detector import MultiMaterialRoofDetector
from .exporter import ResultExporter
from .image_source import GeoTiffImageSource


class RoofAnalysisApplication:
    """Coordinates loading, detection, attribute extraction and export."""

    def __init__(
            self,
            image_source: GeoTiffImageSource,
            detector: MultiMaterialRoofDetector,
            extractor: RoofAttributeExtractor,
            exporter: ResultExporter,
    ) -> None:
        self.image_source = image_source
        self.detector = detector
        self.extractor = extractor
        self.exporter = exporter

    def run(
            self,
            image_path: Path,
            output_dir: Path,
            max_buildings: int,
    ) -> int:
        geo_image = self.image_source.load(image_path)
        candidates = self.detector.detect(
            geo_image.bgr,
            max_buildings=max_buildings,
        )
        results = [
            self.extractor.extract(geo_image, candidate, index)
            for index, candidate in enumerate(candidates, start=1)
        ]
        self.exporter.export(geo_image, results, output_dir)
        return len(results)
