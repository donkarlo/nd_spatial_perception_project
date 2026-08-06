from __future__ import annotations

from pathlib import Path

from roof_analysis.attributes import RoofAttributeExtractor
from roof_analysis.footprints import GeoJsonBuildingFootprintSource
from roof_analysis.image_source import GeoTiffImageSource


def test_hietzing_building_010_solar_array_regression() -> None:
    project_root = Path(__file__).resolve().parents[1]
    image_path = project_root / "scripts" / "data" / "vienna_hietzing.tif"
    footprints_path = (
        project_root
        / "scripts"
        / "data"
        / "vienna_hietzing_buildings.geojson"
    )

    geo_image = GeoTiffImageSource().load(image_path)
    source = GeoJsonBuildingFootprintSource()
    collection = source.load(
        geo_image=geo_image,
        footprints_path=footprints_path,
    )
    candidate = next(
        item
        for item in collection.candidates
        if item.source_id == "way/202815615"
    )
    candidate = source.materialize_mask(geo_image, candidate)

    result = RoofAttributeExtractor().extract(
        geo_image=geo_image,
        candidate=candidate,
        index=10,
        footprint_source_description=collection.source_description,
    )

    assert result.solar_panels_present is True
    assert result.solar_panel_count == 1
    assert result.confidence["solar_panels"] >= 0.75
    assert result.roof_type == "unknown"
    assert result.material == "unknown"
    assert result.confidence["type"] == 0.0
    assert result.confidence["material"] == 0.0
    assert result.visible_condition == "not_assessed_due_to_solar_coverage"
