from __future__ import annotations

from pathlib import Path

from .attributes import RoofAttributeExtractor
from .exporter import ResultExporter
from .footprints import (
    FootprintError,
    GeoJsonBuildingFootprintSource,
    OverpassBuildingFootprintFetcher,
)
from .image_source import GeoTiffImageSource


class RoofAnalysisApplication:
    """Coordinates geodata loading, selection, extraction and export."""

    def __init__(
        self,
        image_source: GeoTiffImageSource,
        footprint_source: GeoJsonBuildingFootprintSource,
        extractor: RoofAttributeExtractor,
        exporter: ResultExporter,
        overpass_fetcher: OverpassBuildingFootprintFetcher,
    ) -> None:
        self.image_source = image_source
        self.footprint_source = footprint_source
        self.extractor = extractor
        self.exporter = exporter
        self.overpass_fetcher = overpass_fetcher

    def run(
        self,
        image_path: Path,
        output_dir: Path,
        footprints_path: Path | None,
        fetch_osm: bool,
        max_buildings: int,
        min_visible_buildings: int,
    ) -> tuple[int, int, Path]:
        geo_image = self.image_source.load(image_path)

        resolved_footprints = footprints_path
        if resolved_footprints is None:
            if not fetch_osm:
                raise FootprintError(
                    "No building-footprint file was provided. Use --footprints "
                    "or enable --fetch-osm."
                )
            resolved_footprints = output_dir / "downloaded_osm_buildings.geojson"
            self.overpass_fetcher.fetch(
                geo_image=geo_image,
                output_path=resolved_footprints,
            )

        collection = self.footprint_source.load(
            geo_image=geo_image,
            footprints_path=resolved_footprints,
        )
        visible_count = len(collection.candidates)
        if visible_count < min_visible_buildings:
            raise FootprintError(
                "The image does not contain enough usable independent buildings.\n"
                f"Required: at least {min_visible_buildings}\n"
                f"Found: {visible_count}\n"
                "Use a larger orthophoto extent or choose a lower-density filter "
                "only if you document that decision."
            )
        if max_buildings > visible_count:
            raise FootprintError(
                f"Requested {max_buildings} roofs, but only {visible_count} "
                "usable independent buildings are available."
            )

        selected = [
            self.footprint_source.materialize_mask(geo_image, candidate)
            for candidate in collection.candidates[:max_buildings]
        ]
        # Stable spatial order makes JSON and overlays easy to compare.
        selected.sort(key=lambda item: (item.centroid_px[1], item.centroid_px[0]))
        results = [
            self.extractor.extract(
                geo_image=geo_image,
                candidate=candidate,
                index=index,
                footprint_source_description=collection.source_description,
            )
            for index, candidate in enumerate(selected, start=1)
        ]
        self.exporter.export(
            geo_image=geo_image,
            all_candidates=collection.candidates,
            selected_candidates=selected,
            results=results,
            output_dir=output_dir,
            footprint_source_path=resolved_footprints,
        )
        return visible_count, len(results), resolved_footprints
