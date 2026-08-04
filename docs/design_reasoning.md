# Design and reasoning

The program uses one georeferenced aerial image because a high-resolution
orthophoto provides the most direct top-down evidence for roof outlines. A
GeoTIFF is required rather than an ordinary JPG because geographic polygons and
square-metre areas require a coordinate reference system and scale. The source
should preferably be a true orthophoto such as Vienna's open orthophoto data.

The pipeline is intentionally small and object-oriented. `GeoTiffImageSource`
loads the RGB bands and georeferencing. `MultiMaterialRoofDetector` creates
candidate masks for warm tiled, neutral/metal and green surfaces, then filters
regions using area, solidity, rectangularity and edge evidence. The detector is
explainable and reproducible, although it is less general than a trained aerial
building-segmentation model. `RoofAttributeExtractor` simplifies each detected
contour, transforms its vertices to longitude/latitude, calculates planimetric
area, and estimates type, material, orientation, solar panels, rooftop
superstructures and visible condition. `ResultExporter` creates the requested
JSON and review overlays.

Recoverability differs by attribute. Polygon and planimetric area are supported
when both segmentation and GeoTIFF georeferencing are good. Material and roof
type can sometimes be inferred from colour, texture, ridge lines and shape, but
these estimates are weaker and therefore have independent confidence scores.
Solar panels and superstructures can be suggested from small rectangular or
contrast-anomaly regions. Exact slope is not recoverable from a single RGB
orthophoto without elevation, stereo, oblique imagery or a 3D model; the program
therefore returns `null` rather than inventing a number.

Confidence values are attribute-specific. Polygon confidence combines detector
score and contour solidity. Area confidence also includes georeferencing
quality. Type, material and object confidences reflect the strength of their
visual evidence. In a production system, high-confidence records can be accepted
automatically, medium-confidence records can be sampled for quality control, and
low-confidence records should be sent to human review.

To scale from ten buildings to thousands, the same application can process
orthophoto tiles independently, merge duplicate polygons at tile boundaries and
store results in a spatial database. The explainable detector can later be
replaced by a trained segmentation implementation without changing the image
source, attribute extractor or exporter interfaces.
