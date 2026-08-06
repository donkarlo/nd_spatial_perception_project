from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin

from .footprints import (
    GeoJsonBuildingFootprintSource,
    OverpassBuildingFootprintFetcher,
)
from .image_source import GeoTiffImageSource

TILE_SIZE = 256
WEB_MERCATOR_HALF_WORLD = 20037508.342789244
TILE_URL = (
    "https://mapsneu.wien.gv.at/basemap/"
    "bmaporthofoto30cm/normal/google3857/{zoom}/{y}/{x}.jpeg"
)


@dataclass(frozen=True, slots=True)
class ViennaSite:
    name: str
    latitude: float
    longitude: float
    description: str


SITES = {
    "donaustadt": ViennaSite(
        name="donaustadt",
        latitude=48.2515,
        longitude=16.4720,
        description="Real suburban Vienna orthophoto sample in Donaustadt.",
    ),
    "hietzing": ViennaSite(
        name="hietzing",
        latitude=48.1788,
        longitude=16.2732,
        description="Real suburban Vienna orthophoto sample in Hietzing.",
    ),
}


class SampleDownloadError(RuntimeError):
    """Raised when a real sample cannot be downloaded or validated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download real basemap.at orthophoto tiles and matching "
            "OpenStreetMap building footprints for two Vienna sites."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Destination directory (default: data)",
    )
    parser.add_argument(
        "--site",
        choices=["all", *SITES.keys()],
        default="all",
        help="Site to download (default: all)",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="WMTS zoom level (default: 19)",
    )
    parser.add_argument(
        "--tiles",
        type=int,
        default=12,
        help="Square tile count per side (default: 12, must be even)",
    )
    parser.add_argument(
        "--minimum-buildings",
        type=int,
        default=20,
        help="Minimum independent buildings required (default: 20)",
    )
    return parser.parse_args()


def lonlat_to_tile(
    longitude: float,
    latitude: float,
    zoom: int,
) -> tuple[int, int]:
    latitude = max(min(latitude, 85.05112878), -85.05112878)
    n = 2**zoom
    x = int(math.floor((longitude + 180.0) / 360.0 * n))
    latitude_rad = math.radians(latitude)
    y = int(
        math.floor(
            (
                1.0
                - math.asinh(math.tan(latitude_rad)) / math.pi
            )
            / 2.0
            * n
        )
    )
    return x, y


def tile_resolution(zoom: int) -> float:
    return (2.0 * WEB_MERCATOR_HALF_WORLD) / (TILE_SIZE * (2**zoom))


def tile_top_left_meters(
    tile_x: int,
    tile_y: int,
    zoom: int,
) -> tuple[float, float]:
    resolution = tile_resolution(zoom)
    min_x = -WEB_MERCATOR_HALF_WORLD + tile_x * TILE_SIZE * resolution
    max_y = WEB_MERCATOR_HALF_WORLD - tile_y * TILE_SIZE * resolution
    return min_x, max_y


def download_tile(
    session: requests.Session,
    zoom: int,
    tile_x: int,
    tile_y: int,
) -> tuple[np.ndarray, str]:
    url = TILE_URL.format(zoom=zoom, x=tile_x, y=tile_y)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            encoded = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (TILE_SIZE, TILE_SIZE):
                raise SampleDownloadError(
                    f"Invalid tile image returned for {url}"
                )
            return image, url
        except (requests.RequestException, SampleDownloadError) as error:
            last_error = error
            time.sleep(1.0 + attempt)
    raise SampleDownloadError(
        f"Could not download orthophoto tile {url}: {last_error}"
    )


def download_site(
    site: ViennaSite,
    output_dir: Path,
    zoom: int,
    tile_count: int,
    minimum_buildings: int,
) -> tuple[Path, Path]:
    if tile_count < 2 or tile_count % 2 != 0:
        raise SampleDownloadError("--tiles must be an even integer of at least 2")
    if not 1 <= zoom <= 19:
        raise SampleDownloadError("--zoom must be between 1 and 19")

    output_dir.mkdir(parents=True, exist_ok=True)
    center_x, center_y = lonlat_to_tile(
        longitude=site.longitude,
        latitude=site.latitude,
        zoom=zoom,
    )
    half = tile_count // 2
    first_x = center_x - half
    first_y = center_y - half

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "roof-analysis-propx-assessment/1.0 "
                "(real open Vienna sample downloader)"
            )
        }
    )

    mosaic = np.zeros(
        (tile_count * TILE_SIZE, tile_count * TILE_SIZE, 3),
        dtype=np.uint8,
    )
    urls: list[str] = []
    for row in range(tile_count):
        for column in range(tile_count):
            tile_x = first_x + column
            tile_y = first_y + row
            tile, url = download_tile(
                session=session,
                zoom=zoom,
                tile_x=tile_x,
                tile_y=tile_y,
            )
            y0 = row * TILE_SIZE
            x0 = column * TILE_SIZE
            mosaic[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = tile
            urls.append(url)
            print(
                f"{site.name}: downloaded tile "
                f"{row * tile_count + column + 1}/{tile_count * tile_count}",
                end="\r",
                flush=True,
            )
    print()

    min_x, max_y = tile_top_left_meters(first_x, first_y, zoom)
    resolution = tile_resolution(zoom)
    transform = from_origin(min_x, max_y, resolution, resolution)
    geotiff_path = output_dir / f"vienna_{site.name}.tif"
    rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        width=rgb.shape[1],
        height=rgb.shape[0],
        count=3,
        dtype=np.uint8,
        crs="EPSG:3857",
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        dataset.write(np.moveaxis(rgb, -1, 0))
        dataset.update_tags(
            SOURCE_DESCRIPTION=(
                "Real basemap.at orthophoto WMTS mosaic; "
                "Datenquelle: basemap.at, CC BY 4.0."
            ),
            SAMPLE_SITE=site.name,
            SAMPLE_DESCRIPTION=site.description,
        )

    geo_image = GeoTiffImageSource().load(geotiff_path)
    footprints_path = output_dir / f"vienna_{site.name}_buildings.geojson"
    OverpassBuildingFootprintFetcher().fetch(
        geo_image=geo_image,
        output_path=footprints_path,
    )

    collection = GeoJsonBuildingFootprintSource().load(
        geo_image=geo_image,
        footprints_path=footprints_path,
    )
    building_count = len(collection.candidates)
    if building_count < minimum_buildings:
        geotiff_path.unlink(missing_ok=True)
        footprints_path.unlink(missing_ok=True)
        raise SampleDownloadError(
            f"Site {site.name} produced only {building_count} usable "
            f"independent buildings; {minimum_buildings} are required. "
            "Increase --tiles or choose another center."
        )

    metadata: dict[str, Any] = {
        "site": site.name,
        "description": site.description,
        "center_wgs84": [site.longitude, site.latitude],
        "zoom": zoom,
        "tile_count_per_side": tile_count,
        "source": "basemap.at Orthofoto WMTS",
        "attribution": "Datenquelle: basemap.at, CC BY 4.0",
        "building_source": "OpenStreetMap contributors via Overpass API",
        "usable_independent_buildings": building_count,
        "tile_urls": urls,
    }
    (output_dir / f"vienna_{site.name}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return geotiff_path, footprints_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    selected_sites = (
        list(SITES.values())
        if args.site == "all"
        else [SITES[args.site]]
    )
    for site in selected_sites:
        geotiff_path, footprints_path = download_site(
            site=site,
            output_dir=output_dir,
            zoom=args.zoom,
            tile_count=args.tiles,
            minimum_buildings=args.minimum_buildings,
        )
        print(f"Created: {geotiff_path}")
        print(f"Created: {footprints_path}")


if __name__ == "__main__":
    main()
