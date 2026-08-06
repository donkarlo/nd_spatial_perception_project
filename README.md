# Roof Analysis
The application fuses a real georeferenced aerial orthophoto with open building
footprints. By default it requires at least 20 usable independent buildings in
the image, selects 10 roofs, extracts conservative visual attributes, and
exports JSON plus review overlays.

No fabricated aerial image or fabricated building footprint is included in this
repository. Two real Vienna samples can be downloaded reproducibly from open
services with the provided script.

## Data sources

The built-in sample downloader uses:

- **basemap.at Orthofoto WMTS** for real aerial imagery in EPSG:3857;
- **OpenStreetMap building ways through Overpass API** for matching open
  building footprints.

Attribution for downloaded orthophotos:

```text
Datenquelle: basemap.at, CC BY 4.0
```

Attribution for building footprints:

```text
© OpenStreetMap contributors
```

The downloader stores the exact tile URLs, bounds, source attribution and
building count in a metadata JSON file next to each sample.

# Data directory

Create two real Vienna samples from open online services with:

```bash
python scripts/download_vienna_samples.py --output-dir data
```

The downloader creates, for each named site:

- a real basemap.at orthophoto mosaic as a georeferenced GeoTIFF;
- OpenStreetMap building footprints for the exact image extent;
- a metadata JSON file containing the bounds, tile URLs and attribution.

Network access is required only for the download step. The roof analysis itself can then run offline from the generated
GeoTIFF and GeoJSON files.

## Why image and footprints are fused

An RGB-only colour detector easily confuses trees, roads, shadows and gravel
with roofs. Open building polygons provide the building location and a stable
outline prior. Pixels inside each polygon are then used to estimate visual
attributes such as material, roof type, orientation, solar panels and visible
superstructures.

This separation is intentional:

- geometry and planimetric area come primarily from the aligned footprint;
- visual attributes come from the orthophoto;
- exact metric slope is not claimed from one RGB image and is exported as
  `null` with confidence `0.0`.

## Project structure

```text
.
├── Dockerfile
├── README.md
├── SOURCE_DATA.md
├── design_reasoning.md
├── requirements.txt
├── run.sh
├── scripts/
│   └── download_vienna_samples.py
├── src/roof_analysis/
│   ├── __main__.py
│   ├── application.py
│   ├── attributes.py
│   ├── exporter.py
│   ├── footprints.py
│   ├── georeferencing.py
│   ├── image_source.py
│   ├── models.py
│   └── sample_download.py
├── data/                         # created by the downloader
└── tests/
```

## Local installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

## Download two real Vienna samples

Network access is required for this step:

```bash
python scripts/download_vienna_samples.py --output-dir data
```

The command downloads two independent, continuous orthophoto mosaics:

```text
data/vienna_donaustadt.tif
data/vienna_donaustadt_buildings.geojson
data/vienna_donaustadt_metadata.json

data/vienna_hietzing.tif
data/vienna_hietzing_buildings.geojson
data/vienna_hietzing_metadata.json
```

Each sample is validated after download. If fewer than 20 usable independent
buildings are present, the downloader removes that sample and reports an error
instead of silently producing an unsuitable fixture.

Download only one site:

```bash
python scripts/download_vienna_samples.py \
  --output-dir data \
  --site donaustadt
```

A larger mosaic can be requested when needed:

```bash
python scripts/download_vienna_samples.py \
  --output-dir data \
  --site donaustadt \
  --tiles 14
```

## Interactive execution

```bash
./run.sh
```

The program asks for:

1. absolute GeoTIFF path;
2. absolute footprint GeoJSON path, or Enter to fetch OSM automatically;
3. absolute output root;
4. number of roofs to select;
5. minimum number of independent buildings required;
6. minimum building area thresholds.

After the run it asks:

```text
Do you want to analyze another image? [y/N]:
```

A JPG, PNG, ordinary TIFF without CRS, or a broken GeoTIFF is rejected with a
clear error before output processing starts.

## Non-interactive execution

Using the downloaded footprint file:

```bash
PYTHONPATH=src python -m roof_analysis \
  "$(pwd)/data/vienna_donaustadt.tif" \
  --footprints "$(pwd)/data/vienna_donaustadt_buildings.geojson" \
  --output-root "$(pwd)/outputs" \
  --max-buildings 10 \
  --min-visible-buildings 20
```

Or fetch OSM footprints automatically from the GeoTIFF bounds:

```bash
PYTHONPATH=src python -m roof_analysis \
  "$(pwd)/data/vienna_donaustadt.tif" \
  --fetch-osm \
  --output-root "$(pwd)/outputs"
```

## Docker

Build:

```bash
docker build -t roof_analysis .
```

Interactive execution:

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  roof_analysis
```

Inside the container use paths such as:

```text
/app/data/vienna_donaustadt.tif
/app/data/vienna_donaustadt_buildings.geojson
/app/outputs
```

Non-interactive execution:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  roof_analysis \
  /app/data/vienna_donaustadt.tif \
  --footprints /app/data/vienna_donaustadt_buildings.geojson \
  --output-root /app/outputs
```

Downloading samples from Docker requires a writable data mount and network
access:

```bash
docker run --rm \
  --entrypoint python \
  -v "$(pwd)/data:/app/data" \
  roof_analysis \
  -m roof_analysis.sample_download \
  --output-dir /app/data
```

## Output structure

Every run replaces only the result directory for the current image name:

```text
outputs/
└── vienna_donaustadt/
    ├── roof_attributes.json
    ├── downloaded_osm_buildings.geojson   # only with --fetch-osm
    └── overlays/
        ├── 00_all_roofs_overlay.jpg
        ├── 01_building_001.jpg
        ├── 02_building_002.jpg
        ├── 03_building_003.jpg
        └── 04_building_004.jpg
```

In the overview image:

- thin grey outlines show every usable independent building;
- thick green outlines show the 10 selected roofs;
- red labels identify the exported records.

The JSON contains the input paths, total usable-building count, selected count,
and one structured record per selected roof.

## Confidence scores

The exported values are deterministic evidence-based scores in `[0, 1]`, not
statistical confidence intervals and not calibrated probabilities. The task asks
for per-attribute confidence scores; producing statistical confidence intervals
would require a labelled validation population and an explicit probabilistic
error model.

Two common quantities are reused by several attributes:

- `boundary_support` is the fraction of the rasterised footprint boundary that
  lies on, or very close to, an image edge. The implementation draws the
  footprint boundary, dilates the Canny edge map with a 7x7 kernel and measures
  the fraction of boundary pixels supported by those nearby image edges.
- `image_quality` is

  `0.45*texture_score + 0.30*brightness_score + 0.25*clipping_score`

  where `texture_score = clip(gray_std/45, 0, 1)`,
  `brightness_score = clip(1 - abs(mean_gray-135)/135, 0, 1)`, and
  `clipping_score = clip(1 - 4*clipped_fraction, 0, 1)`. `clipped_fraction` is
  the fraction of roof pixels with grayscale values `<=8` or `>=247`.

Most visual attributes then use a common image-quality multiplier:

`quality_factor = 0.68 + 0.32*image_quality`

and

`visual_confidence = clip(attribute_evidence * quality_factor, 0, attribute_cap)`.

The polygon itself is copied from the aligned source footprint, so a separate
`polygon` confidence is intentionally not exported.

### `area`

Area confidence combines footprint-to-image boundary agreement, image quality
and raster scale:

`scale_support = ramp(log(1+pixel_area), log(301), log(10001))`

`area_confidence = clip(0.46 + 0.18*boundary_support + 0.16*image_quality + 0.12*scale_support, 0, 0.93)`.

The logarithmic scale term rewards roofs represented by more raster pixels
without allowing very large buildings to dominate. The score remains below
`1.0` because an OSM building footprint can differ from the visible roof
overhang.

### `type`

Roof-type evidence is derived from Canny edges and Hough line segments inside
an eroded roof mask. Lines are grouped into 15-degree orientation bins.
`line_support` measures total accepted line length relative to footprint
perimeter, and `dominant_share`, `second_share` and `third_share` describe how
the line evidence is distributed between angle bins.

If no accepted line exists, the roof is labelled `flat_or_low_pitch` and:

`type_evidence = 0.30 + 0.28*(1 - ramp(edge_density, 0.025, 0.13))`.

For `complex` roofs:

`type_evidence = 0.32 + 0.24*line_support + 0.22*min(1,(second_share+third_share)/0.45) + 0.12*ramp(edge_density,0.04,0.16)`.

For `hipped` roofs:

`type_evidence = 0.34 + 0.22*line_support + 0.18*separation_support + 0.16*balance`.

For `pitched` roofs:

`type_evidence = 0.34 + 0.25*line_support + 0.22*dominant_share + 0.10*max(0,dominant_share-second_share)`.

The final value is:

`type_confidence = clip(type_evidence * quality_factor, 0, 0.84)`.

### `orientation`

When Hough-line evidence exists, the dominant roof-line angle is used.
`orientation_evidence` is:

`clip(0.28 + 0.32*angular_coherence + 0.22*line_support + 0.12*dominant_share, 0, 0.88)`.

If no reliable roof-line direction is available, orientation falls back to the
minimum-area rectangle of the footprint. Its weaker evidence is:

`clip(0.22 + 0.38*elongation_evidence + 0.18*rectangularity, 0, 0.72)`.

The final score is:

`orientation_confidence = clip(0.58*orientation_evidence + 0.18*boundary_support + 0.12*image_quality + ridge_bonus, 0, 0.86)`

where `ridge_bonus = 0.08` only when the orientation comes from image line
evidence and `0.0` for the footprint fallback.

### `material`

The detector creates competing scores for `tiled`, `green_roof`, `metal`,
`flat/gravel` and `glass` from median HSV colour, red/green pixel fractions and
grayscale texture. Let `best` be the highest material score, `second` the next
highest and `margin = max(0,best-second)`.

If `best < 0.38`, the result is `unknown` and:

`material_evidence = clip(0.16 + 0.18*(1-best), 0, 0.34)`.

Otherwise:

`material_evidence = clip(0.34 + 0.38*best + 0.24*margin, 0, 0.88)`.

The implementation then deliberately downweights this relatively uncertain
attribute:

`material_confidence = clip((0.86*material_evidence) * quality_factor, 0, 0.80)`.

### `solar_panels`

The roof is first rectified using its minimum-area rectangle. The solar grid
evidence combines five signals:

`grid_evidence = 0.42*periodicity_support + 0.18*low_saturation_support + 0.18*edge_density_support + 0.12*dark_neutral_support + 0.10*line_count_support`.

The supports are normalized with the following ranges:

- periodicity: `0.22 -> 0.52`;
- median saturation: inverse ramp `18 -> 70`;
- edge density: `0.16 -> 0.34`;
- dark-neutral fraction: `0.45 -> 0.90`;
- Hough-line count: `20 -> 55`.

If `grid_evidence < 0.72`, solar panels are reported absent and the evidence in
that negative decision is:

`absence_evidence = 0.22 + 0.42*(1-grid_evidence)`.

If `grid_evidence >= 0.72` but the coherent dark panel mask covers less than
`15%` of the roof, the result is still negative with:

`uncertain_absence = 0.18 + 0.24*(1-grid_evidence)`.

For a positive detection:

`positive_evidence = 0.46 + 0.45*grid_evidence`.

The selected evidence is multiplied by `quality_factor` and capped at `0.86`.

At the current orthophoto resolution the implementation reports a binary solar
count (`0` or `1`): zero means no array was accepted and one means solar-panel
presence was accepted. It does **not** reliably count individual modules or
multiple separate arrays.

### `superstructures`

This output is a count of candidate rooftop objects, not semantic identities
such as "chimney" or "HVAC unit". A 17x17 local grayscale mean is subtracted
from the roof image, pixels with deviation above `34` are thresholded, and
connected regions with an accepted area are counted.

For a positive count:

`superstructure_evidence = 0.30 + 0.20*ramp(count,1,8) + 0.20*ramp(object_area_share,0.003,0.08) + 0.14*band_score(contrast_fraction,0.01,0.16,0.12)`.

For zero accepted candidates:

`superstructure_evidence = 0.26 + 0.24*(1-ramp(contrast_fraction,0.01,0.16))`.

The final score is:

`superstructure_confidence = clip(superstructure_evidence * quality_factor, 0, 0.78)`.

### `visible_condition`

This is deliberately a coarse visible-surface assessment. The detector computes
the fraction of pixels darker than `40`, the fraction brighter than `235`, their
sum (`extreme_fraction`) and grayscale standard deviation (`texture`).

`contrast_risk = ramp(extreme_fraction,0.08,0.32)`

`texture_risk = ramp(texture,35,75)`.

If `extreme_fraction > 0.20` or `texture > 62`, the label is
`partly_obscured_or_high_contrast` and:

`condition_evidence = 0.30 + 0.30*max(contrast_risk,texture_risk)`.

Otherwise the label is `no_obvious_issue_visible` and:

`clean_visibility = 1 - max(contrast_risk,0.65*texture_risk)`

`condition_evidence = 0.28 + 0.30*clean_visibility`.

The final score is image-quality weighted and capped at `0.68`. The low cap is
intentional because an RGB orthophoto cannot establish structural roof
condition.

### `slope`

`slope_deg` is `null` and slope confidence is always `0.0`. A single RGB
orthophoto does not contain defensible metric height information, so the program
does not invent a slope estimate.

### Solar-dominated roofs

If solar panels are detected and either panel coverage is at least `45%` or the
remaining non-panel roof pixels are below `30%`, the panel grid is considered to
dominate the visible roof. In that case `type`, underlying `material`,
`superstructures` and `visible_condition` are not inferred from panel edges.
They are exported as unknown/not assessed with confidence `0.0`.

## Known limitations

- OSM completeness and alignment vary by location and date.
- The automatic OSM fetcher intentionally reads building **ways**; complex
  multipolygon relations may require a separately exported GeoJSON source.
- A building footprint is not always identical to the visible roof overhang.
- Trees and shadows can reduce attribute confidence even when geometry is good.
- Roof type, material, condition and solar-panel detection are explainable
  heuristics, not a trained universal segmentation model.
- The program deliberately fails when fewer than 20 independent buildings are
  available instead of pretending that the requested comparison set exists.

## What would be improved with more time

- use Vienna's official high-resolution True Orthophoto download directly;
- fuse the official Vienna building model or roof model instead of relying only
  on OSM ways;
- add a learned roof/material model with a labelled validation set;
- use DSM/LiDAR or the LOD2.1 roof model for metric slope and stronger type
  classification;
- quantify footprint-to-image alignment and optionally refine polygons against
  roof edges;
- add caching and bounded concurrency for large-area processing.
