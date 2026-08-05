# Rooftop Detection and Attribute Extraction

A small object-oriented Python 3.13 application for the PropX technical assessment. It reads a georeferenced aerial
GeoTIFF, detects roof candidates, extracts roof attributes with confidence scores, and exports the results as JSON and
overlay images.

## Why the input must be a GeoTIFF

A normal JPG or PNG contains pixels but normally has no map coordinates or physical scale. The required output contains
roof polygons in longitude and latitude and roof areas in square metres. The input must therefore provide georeferencing
information.

This implementation accepts a GeoTIFF containing at least three image bands, a coordinate reference system, and either
an affine transform or ground-control points.

## Run with Docker

Build the Docker image from the project root:

```bash
docker build -t roof_analysis .
```

### Interactive execution with the included sample

Run the container with an interactive terminal and mount the output directory:

```bash
docker run --rm -it \
  -v "$(pwd)/outputs:/app/outputs" \
  roof_analysis
```

The program asks for:

- the GeoTIFF image path;
- the output directory;
- the maximum number of buildings;
- the minimum roof area in pixels.

Press Enter to accept the displayed default values.

Inside Docker, the included sample image is available at:

```text
/app/data/sample_vienna.tif
```

After each analysis, the program asks:

```text
Do you want to analyze another image? [y/N]:
```

Enter `y` to process another image or press Enter to finish.

### Interactive execution with another GeoTIFF

Mount the directory containing the input GeoTIFF as a read-only Docker volume:

```bash
docker run --rm -it \
  -v "/absolute/path/to/input-directory:/app/input:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  roof_analysis
```

When the program asks for the image path, enter the path inside the container, for example:

```text
/app/input/orthophoto.tif
```

A host path such as:

```text
/home/user/images/orthophoto.tif
```

is not automatically visible inside the container. The input directory must first be mounted as a Docker volume, as
shown above.

### Non-interactive Docker execution

Arguments may also be supplied directly for a one-time execution:

```bash
docker run --rm \
  -v "/absolute/path/to/input-directory:/app/input:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  roof_analysis \
  /app/input/orthophoto.tif \
  --output /app/outputs \
  --max-buildings 10 \
  --min-area-px 450
```

## Run without Docker

Create and activate a Python 3.13 virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Interactive execution

Run the application without arguments:

```bash
python src/nd_spatial_perception/__main__.py
```

The application asks for the input path and processing options and offers to process another image after each run.

### Non-interactive execution

Run one analysis by supplying the input path and options directly:

```bash
python src/nd_spatial_perception/__main__.py \
  data/sample_vienna.tif \
  --output outputs \
  --max-buildings 10 \
  --min-area-px 450
```

For another image:

```bash
python src/nd_spatial_perception/__main__.py \
  /absolute/path/to/orthophoto.tif \
  --output outputs
```

The default values for omitted optional arguments are:

```text
output directory: outputs
maximum buildings: 10
minimum roof area: 450 pixels
```

## Outputs

The application creates the following structure:

```text
outputs/
├── roof_attributes.json
└── overlays/
    ├── 00_all_roofs_overlay.jpg
    ├── 01_building_001.jpg
    ├── 02_building_002.jpg
    └── ...
```

Each JSON record follows this structure:

```json
{
  "building_id": "building_001",
  "source_used": "...",
  "roof": {
    "polygon": [
      [16.0, 48.0]
    ],
    "area_m2": 0.0,
    "type": "pitched",
    "material": "tiled",
    "orientation": "NE-SW",
    "orientation_deg": 45.0,
    "slope_deg": null,
    "solar_panels": {
      "present": false,
      "count": 0
    },
    "superstructures": {
      "count": 0
    },
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

- `GeoTiffImageSource` reads the image and its georeferencing.
- `MultiMaterialRoofDetector` finds roof-like regions from colour, shape, and edge evidence.
- `RoofAttributeExtractor` calculates polygons, areas, and visual attributes.
- `ResultExporter` writes JSON and overlay images.
- `RoofAnalysisApplication` coordinates loading, detection, extraction, and export.

## Source selection and trade-offs

The implementation uses a high-resolution aerial RGB GeoTIFF because a top-down orthophoto provides direct roof
visibility and georeferencing for polygon and area calculation.

It is easier to process reproducibly than street-level or oblique imagery and provides substantially more building-level
detail than medium-resolution satellite imagery.

The main limitation is that one RGB orthophoto contains no direct elevation measurement. It therefore cannot support a
reliable metric roof slope.

Material, roof type, solar panels, superstructures, and visible condition are visual estimates and receive independent
confidence scores.

## What is and is not recoverable

The roof polygon and planimetric area are derived from image segmentation and GeoTIFF georeferencing.

Roof type, material, orientation, solar panels, superstructures, and visible condition are heuristic visual estimates.

Exact roof slope is not defensible from one RGB orthophoto without an elevation source such as LiDAR, a digital surface
model, stereo imagery, or a 3D city model.

For that reason, `slope_deg` is `null` and its confidence is `0.0`.

## Important limitation

This is an explainable computer-vision baseline rather than a trained universal building-segmentation model.

It works best on high-resolution true orthophotos with clearly visible roofs. Low-confidence results should be manually
reviewed.

The included sample is a real Vienna aerial image with approximate demonstration georeferencing. A true orthophoto
GeoTIFF should be used for metric production results.