# Rooftop Detection and Attribute Extraction

A small object-oriented Python 3.13 program for the PropX technical assessment.
It accepts **one georeferenced aerial GeoTIFF path**, detects up to ten roof
candidates, and writes one JSON record per detected building.

## Why the input must be a GeoTIFF

A normal JPG or PNG contains pixels but normally has no map coordinates or
physical scale. The required output contains `[[lon, lat], ...]` and `area_m2`,
so the input must contain georeferencing. This implementation therefore accepts
a GeoTIFF with a CRS and either an affine transform or ground-control points.

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py data/sample_vienna.tif --output outputs --max-buildings 10
```

For another image:

```bash
python main.py /absolute/path/to/orthophoto.tif --output outputs
```

## Outputs

```text
outputs/
├── roof_attributes.json
└── overlays/
    ├── 00_all_roofs_overlay.jpg
    ├── 01_building_001.jpg
    ├── 02_building_002.jpg
    ├── 03_building_003.jpg
    └── 04_building_004.jpg
```

Each JSON record follows this structure:

```json
{
  "building_id": "building_001",
  "source_used": "...",
  "roof": {
    "polygon": [[16.0, 48.0]],
    "area_m2": 0.0,
    "type": "pitched",
    "material": "tiled",
    "orientation": "NE-SW",
    "orientation_deg": 45.0,
    "slope_deg": null,
    "solar_panels": {"present": false, "count": 0},
    "superstructures": {"count": 0},
    "visible_condition": "no_obvious_issue_visible"
  },
  "confidence": {
    "polygon": 0.0,
    "area": 0.0,
    "type": 0.0,
    "material": 0.0,
    "orientation": 0.0,
    "slope": 0.0,
    "solar_panels": 0.0,
    "superstructures": 0.0,
    "visible_condition": 0.0
  },
  "notes": "..."
}
```

## Object-oriented design

- `GeoTiffImageSource`: reads the image and its georeferencing.
- `MultiMaterialRoofDetector`: finds roof-like regions from colour, shape and edge evidence.
- `RoofAttributeExtractor`: calculates polygon, area and visual attributes.
- `ResultExporter`: writes JSON and overlay images.
- `RoofAnalysisApplication`: coordinates the four steps.

## What is and is not recoverable from one RGB orthophoto

The polygon and planimetric area are derived from image segmentation and the
GeoTIFF georeferencing. Roof type, material, solar panels, superstructures and
visible condition are heuristic visual estimates and receive separate confidence
scores. Exact roof slope is not defensible from one RGB orthophoto without an
elevation source, so `slope_deg` is `null` and its confidence is `0.0`.

## Important limitation

This is an explainable computer-vision baseline rather than a trained universal
building-segmentation model. It works best on high-resolution true orthophotos
with clearly visible roofs. Low-confidence results should be manually reviewed.
The included sample is a real Vienna aerial image with approximate demonstration
georeferencing; use a true orthophoto GeoTIFF for metric production results.
