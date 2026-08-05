# Rooftop Detection and Attribute Extraction

A small object-oriented Python 3.13 application for the PropX technical assessment. It reads one georeferenced aerial GeoTIFF, detects roof candidates, extracts roof attributes with confidence scores, and exports the results as JSON and overlay images.

## Input requirement

The input must be a GeoTIFF containing at least three image bands, a coordinate reference system, and either an affine transform or ground-control points. A normal JPG or PNG is not accepted because it normally contains no geographic coordinates or physical scale.

## Build with Docker

Run this command once from the project root:

```bash
docker build --no-cache -t roof_analysis .
```

Later builds may use:

```bash
docker build -t roof_analysis .
```

## Run

Run the application with:

```bash
./run.sh
```

The program asks for all processing values:

```text
Absolute GeoTIFF image path [/absolute/project/path/data/sample_vienna.tif]:
Absolute output root directory [/absolute/project/path/outputs]:
Maximum number of buildings [10]:
Minimum roof area in pixels [450]:
```

Press Enter to accept a displayed default value.

After the analysis, the program asks:

```text
Do you want to analyze another image? [y/N]:
```

Enter `y` to process another GeoTIFF or press Enter to exit.

The image path must be absolute. For example:

```text
/home/donkarlo/Dropbox/repo/nd_spatial_perception_project/data/sample_vienna.tif
```

## Direct Docker command

`run.sh` executes this command internally:

```bash
docker run --rm -it \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  roof_analysis
```

The project is mounted at the same absolute path inside the container. Therefore, the same absolute image path is valid both on Ubuntu and inside Docker.

## Output structure

The selected output directory is treated as an output root. For every input image, the application creates a child directory whose name is the input filename without `.tif` or `.tiff`.

For this input:

```text
/home/donkarlo/Dropbox/repo/nd_spatial_perception_project/data/sample_vienna.tif
```

and the default output root, the result is:

```text
outputs/
└── sample_vienna/
    ├── roof_attributes.json
    └── overlays/
        ├── 00_all_roofs_overlay.jpg
        ├── 01_building_001.jpg
        ├── 02_building_002.jpg
        └── ...
```

Before processing an image, only that image's existing output directory is deleted and recreated. Therefore, running the same filename again works normally and produces a completely new JSON file and new overlay images. Other image result directories inside `outputs/` are not deleted.

For example:

```text
outputs/
├── sample_vienna/
├── vienna_orthophoto/
└── graz_orthophoto/
```

Re-running `sample_vienna.tif` replaces only `outputs/sample_vienna/`.

## Non-interactive Docker execution

The absolute image path can be supplied directly:

```bash
docker run --rm \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  roof_analysis \
  "$PWD/data/sample_vienna.tif"
```

Optional values can also be supplied:

```bash
docker run --rm \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  roof_analysis \
  "$PWD/data/sample_vienna.tif" \
  --output "$PWD/outputs" \
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

Run interactively:

```bash
python src/nd_spatial_perception/__main__.py
```

Or provide an absolute image path directly:

```bash
python src/nd_spatial_perception/__main__.py \
  "$PWD/data/sample_vienna.tif"
```

## Object-oriented design

- `GeoTiffImageSource` reads and validates the image and its georeferencing.
- `MultiMaterialRoofDetector` detects roof-like regions.
- `RoofAttributeExtractor` extracts polygons, areas, and visual attributes.
- `ResultExporter` writes JSON and overlay images.
- `RoofAnalysisApplication` coordinates loading, detection, extraction, and export.

## Source selection and trade-offs

The implementation uses a high-resolution aerial RGB GeoTIFF because a top-down orthophoto provides direct roof visibility and georeferencing for polygon and area calculation. It provides more building-level detail than medium-resolution satellite imagery and is easier to process reproducibly than street-level or oblique imagery.

A single RGB orthophoto contains no direct elevation information. Therefore, exact metric roof slope cannot be calculated reliably. Roof type, material, orientation, solar panels, superstructures, and visible condition are heuristic visual estimates and receive separate confidence scores.

## Important limitation

This implementation is an explainable computer-vision baseline rather than a trained universal building-segmentation model. It works best with high-resolution true orthophotos containing clearly visible roofs. Low-confidence results should be manually reviewed.

The included sample contains approximate demonstration georeferencing. A true orthophoto GeoTIFF should be used for metric production results.
