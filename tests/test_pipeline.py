from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from roof_analysis.application import RoofAnalysisApplication
from roof_analysis.attributes import RoofAttributeExtractor
from roof_analysis.exporter import ResultExporter
from roof_analysis.footprints import (
    GeoJsonBuildingFootprintSource,
    OverpassBuildingFootprintFetcher,
)
from roof_analysis.image_source import GeoTiffImageSource


def create_test_data(root: Path) -> tuple[Path, Path]:
    width = 900
    height = 900
    image = np.full((height, width, 3), 115, dtype=np.uint8)
    features = []
    building_index = 0

    for row in range(5):
        for column in range(5):
            building_index += 1
            x0 = 55 + column * 165
            y0 = 55 + row * 165
            x1 = x0 + 78
            y1 = y0 + 58
            color = (
                65 + (building_index * 7) % 80,
                70 + (building_index * 11) % 80,
                135 + (building_index * 13) % 90,
            )
            cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
            cv2.line(image, (x0, y0), (x1, y1), (230, 230, 230), 2)
            cv2.line(image, (x1, y0), (x0, y1), (40, 40, 40), 2)

            # from_origin(0, 1000, 1, 1): world y = 1000 - pixel y
            coordinates = [
                [float(x0), float(1000 - y0)],
                [float(x1), float(1000 - y0)],
                [float(x1), float(1000 - y1)],
                [float(x0), float(1000 - y1)],
                [float(x0), float(1000 - y0)],
            ]
            features.append(
                {
                    "type": "Feature",
                    "id": f"building-{building_index}",
                    "properties": {
                        "id": f"building-{building_index}",
                        "building": "yes",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coordinates],
                    },
                }
            )

    geotiff_path = root / "test_city.tif"
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype=np.uint8,
        crs="EPSG:3857",
        transform=from_origin(0.0, 1000.0, 1.0, 1.0),
    ) as dataset:
        dataset.write(np.moveaxis(rgb, -1, 0))

    footprints_path = root / "test_city_buildings.geojson"
    footprints_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": "Test building footprints",
                "crs": {
                    "type": "name",
                    "properties": {"name": "EPSG:3857"},
                },
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    return geotiff_path, footprints_path


def test_pipeline_requires_20_and_exports_10(tmp_path: Path) -> None:
    geotiff_path, footprints_path = create_test_data(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    application = RoofAnalysisApplication(
        image_source=GeoTiffImageSource(),
        footprint_source=GeoJsonBuildingFootprintSource(
            min_area_m2=20.0,
            min_area_px=300.0,
        ),
        extractor=RoofAttributeExtractor(),
        exporter=ResultExporter(),
        overpass_fetcher=OverpassBuildingFootprintFetcher(),
    )
    visible_count, selected_count, used_path = application.run(
        image_path=geotiff_path,
        output_dir=output_dir,
        footprints_path=footprints_path,
        fetch_osm=False,
        max_buildings=10,
        min_visible_buildings=20,
    )

    assert visible_count == 25
    assert selected_count == 10
    assert used_path == footprints_path
    assert (output_dir / "roof_attributes.json").exists()
    assert (output_dir / "overlays" / "00_all_roofs_overlay.jpg").exists()

    result = json.loads(
        (output_dir / "roof_attributes.json").read_text(encoding="utf-8")
    )
    assert result["summary"]["usable_independent_buildings"] == 25
    assert result["summary"]["selected_roofs"] == 10
    assert len(result["buildings"]) == 10

    expected_confidence_keys = {
        "area",
        "type",
        "material",
        "orientation",
        "slope",
        "solar_panels",
        "superstructures",
        "visible_condition",
    }
    for building in result["buildings"]:
        confidence = building["confidence"]
        assert set(confidence) == expected_confidence_keys
        assert "polygon" not in confidence
        assert confidence["slope"] == 0.0
        assert all(0.0 <= value <= 1.0 for value in confidence.values())
