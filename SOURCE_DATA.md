# Source data and licensing

## Orthophoto samples

The sample downloader requests real image tiles from the basemap.at Orthofoto
WMTS endpoint:

```text
https://mapsneu.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg
```

Attribution:

```text
Datenquelle: basemap.at
```

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

Project information:

```text
https://basemap.at/orthofoto/
https://basemap.at/#lizenz
```

## Building footprints

Building ways are requested from OpenStreetMap through public Overpass API
instances. The generated GeoJSON records the source in its metadata.

Attribution:

```text
© OpenStreetMap contributors
```

Licence and copyright information:

```text
https://www.openstreetmap.org/copyright
```

## Important provenance rule

Do not replace the downloaded samples with a photocomposite, pasted roof crops,
synthetic building layout or manually invented coordinates. Synthetic images
may be generated temporarily inside unit tests, but must not be presented as
assessment data or committed under `data/` or `outputs/`.
