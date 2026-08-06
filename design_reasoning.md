# Design and reasoning

## Source choice

The component fuses a real vertical orthophoto with open building footprints.
The orthophoto provides direct roof appearance, while the footprints provide a
stable georeferenced building prior. This is more defensible than image-only
colour segmentation because vegetation, roads, shadows and gravel can share
colour ranges with roof surfaces. The sample downloader uses basemap.at
Orthofoto in EPSG:3857 and OpenStreetMap building ways from Overpass. Both
sources are open, cover Vienna and can be acquired reproducibly. The exact tile
requests and source attribution are stored with each downloaded sample.

## Recoverable attributes

The OpenStreetMap polygon is used as a georeferenced proxy for the building/roof
plan outline; it is not claimed to be an independently detected roof boundary.
It supports planimetric area estimation and provides a useful fallback
orientation prior. The orthophoto supports conservative estimates of roof
material, roof type, ridge orientation and solar-panel presence, plus candidate
rooftop superstructures and coarse visible surface irregularities. Exact metric
roof slope is not recoverable from a single RGB orthophoto, so `slope_deg` is
`null` and slope confidence is `0.0`. A separate polygon confidence is not
exported because the polygon itself comes from the aligned source footprint;
the image-to-footprint alignment evidence is instead included in the area and
orientation confidence scores.

## Alignment and selection

The GeoTIFF CRS and affine transform map image pixels to world coordinates. The
GeoJSON polygons are reprojected to the raster CRS, transformed to pixels and
rasterised as masks. Buildings clipped by the image boundary, too small in
metres or pixels, touching/overlapping another footprint, or closer than the
configured independence gap are rejected. By default the application requires
at least 20 usable independent buildings, ranks them using boundary support,
image quality, area and compactness, and exports the best 10 in stable spatial
order. The overview image shows all usable footprints in grey and the selected
roofs in green so the selection can be reviewed visually.

## Confidence scoring

The exported values are deterministic evidence-based confidence scores in
`[0, 1]`; they are not statistical confidence intervals and they are not
calibrated probabilities. A labelled validation set would be required to turn
them into empirically calibrated probabilities. The implementation combines
attribute-specific image evidence with a roof-level image-quality factor. The
common image-quality factor rewards usable texture, mid-range exposure and few
clipped black/white pixels.

`area` confidence combines three independent supports: footprint boundary
agreement with nearby image edges, roof image quality and raster scale. The
implemented score is

`clip(0.46 + 0.18*boundary_support + 0.16*image_quality + 0.12*scale_support, 0, 0.93)`,

where `scale_support` increases logarithmically from a 300-pixel footprint to a
10,000-pixel footprint.

`type` confidence is derived from Canny edges and Hough line segments inside the
roof mask. The evidence depends on the total line length relative to the roof
perimeter, the share of line energy in dominant 15-degree angle bins, angular
separation between strong directions, balance between those directions and edge
density. The formula changes with the selected class (`flat_or_low_pitch`,
`pitched`, `hipped` or `complex`) and the resulting evidence is multiplied by
the image-quality factor and capped at `0.84`.

`orientation` first uses the dominant roof-line angle when reliable Hough-line
evidence exists. Its evidence combines angular coherence, line support and the
dominant-angle share. If no ridge/line direction is available, the method falls
back to the minimum-area rectangle of the footprint and uses its elongation and
rectangularity as weaker evidence. Final orientation confidence combines
`0.58*orientation_evidence + 0.18*boundary_support + 0.12*image_quality`, with an
additional `0.08` only when the estimate is ridge-based, and is capped at
`0.86`.

`material` confidence comes from competing material hypotheses computed from
median HSV colour, red/green pixel fractions and grayscale texture. For the
winning hypothesis, confidence increases with both its evidence and its margin
over the second-best hypothesis. If no hypothesis reaches the minimum evidence
threshold, the material is reported as `unknown` with a low confidence. The
material evidence is then image-quality weighted and capped at `0.80`.

`solar_panels` confidence uses a rectified roof patch and combines five signals:
repeated row/column grid periodicity (`42%`), low median saturation (`18%`),
edge density (`18%`), dark neutral surface fraction (`12%`) and Hough line count
(`10%`). Grid evidence below `0.72` produces a confidence in absence that falls
as the evidence approaches the threshold. Above that threshold, at least `15%`
coherent dark-surface coverage is also required before solar presence is
reported. Positive evidence is `0.46 + 0.45*grid_evidence`, then image-quality
weighted and capped at `0.86`. At this resolution the implementation reports a
binary array count (`0` or `1`); it does not reliably count individual modules
or multiple separate arrays.

`superstructures` confidence refers to candidate rooftop objects, not semantic
object identities. The detector compares each roof pixel with a local 17x17
mean, thresholds strong local contrast, and counts connected candidate regions
within the allowed area range. For a positive count, confidence increases with
candidate count, occupied roof-area share and a plausible overall
high-contrast fraction; for zero candidates, confidence in absence increases
when the roof contains little high-contrast clutter. The result is
image-quality weighted and capped at `0.78`.

`visible_condition` is intentionally a coarse visual assessment. It uses the
fractions of very dark and very bright pixels together with grayscale texture.
High extreme-pixel fraction or high texture yields
`partly_obscured_or_high_contrast`; otherwise the result is
`no_obvious_issue_visible`. Its confidence is image-quality weighted and capped
conservatively because an RGB orthophoto cannot establish structural condition.

`slope` confidence is always `0.0` because metric roof slope is not inferred
from one RGB orthophoto.

When a detected solar surface dominates the usable roof area (`>=45%` panel
coverage or `<30%` residual roof pixels), the underlying `type`, `material`,
`superstructures` and `visible_condition` are not guessed from the panel grid;
they are reported as unknown/not assessed with confidence `0.0`.

## Scaling

For thousands of buildings, orthophoto and footprint acquisition would be
cached by spatial tile. Candidate polygons could be processed independently in
parallel workers. The current O(n²) independence check would be replaced with a
spatial index. Results would be written to a spatial database with source date,
model version and confidence fields. Low-confidence records would enter a human
review queue. DSM/LiDAR or Vienna's LOD2.1 roof model would be fused for metric
slope and stronger roof-type estimation.
