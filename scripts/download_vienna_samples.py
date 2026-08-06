#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow the script to run directly without installing the package first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin

from roof_analysis.footprints import (
    FootprintError,
    GeoJsonBuildingFootprintSource,
    OverpassBuildingFootprintFetcher,
)
from roof_analysis.image_source import (
    GeoTiffImageSource,
    GeoTiffValidationError,
)

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


SITES: dict[str, ViennaSite] = {
    "donaustadt": ViennaSite(
        name="donaustadt",
        latitude=48.2515,
        longitude=16.4720,
        description=(
            "Real suburban Vienna orthophoto sample in Donaustadt."
        ),
    ),
    "hietzing": ViennaSite(
        name="hietzing",
        latitude=48.1788,
        longitude=16.2732,
        description=(
            "Real suburban Vienna orthophoto sample in Hietzing."
        ),
    ),
}


class SampleDownloadError(RuntimeError):
    """Raised when a complete sample cannot be downloaded or validated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download real Vienna orthophoto tiles and matching "
            "OpenStreetMap building footprints. A site is published "
            "only after all output files have been created and validated."
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
        help=(
            "Square tile count per side "
            "(default: 12; must be even)"
        ),
    )

    parser.add_argument(
        "--minimum-buildings",
        type=int,
        default=20,
        help=(
            "Minimum usable independent buildings required "
            "(default: 20)"
        ),
    )

    return parser.parse_args()


def validate_options(
        zoom: int,
        tile_count: int,
        minimum_buildings: int,
) -> None:
    if not 1 <= zoom <= 19:
        raise SampleDownloadError(
            "--zoom must be between 1 and 19"
        )

    if tile_count < 2 or tile_count % 2 != 0:
        raise SampleDownloadError(
            "--tiles must be an even integer "
            "greater than or equal to 2"
        )

    if minimum_buildings < 1:
        raise SampleDownloadError(
            "--minimum-buildings must be "
            "greater than or equal to 1"
        )


def lonlat_to_tile(
        longitude: float,
        latitude: float,
        zoom: int,
) -> tuple[int, int]:
    latitude = max(
        min(latitude, 85.05112878),
        -85.05112878,
    )

    tile_count = 2 ** zoom

    tile_x = int(
        math.floor(
            (longitude + 180.0)
            / 360.0
            * tile_count
        )
    )

    latitude_rad = math.radians(latitude)

    tile_y = int(
        math.floor(
            (
                    1.0
                    - math.asinh(
                math.tan(latitude_rad)
            )
                    / math.pi
            )
            / 2.0
            * tile_count
        )
    )

    return tile_x, tile_y


def tile_resolution(zoom: int) -> float:
    return (
            2.0 * WEB_MERCATOR_HALF_WORLD
    ) / (
            TILE_SIZE * (2 ** zoom)
    )


def tile_top_left_meters(
        tile_x: int,
        tile_y: int,
        zoom: int,
) -> tuple[float, float]:
    resolution = tile_resolution(zoom)

    min_x = (
            -WEB_MERCATOR_HALF_WORLD
            + tile_x * TILE_SIZE * resolution
    )

    max_y = (
            WEB_MERCATOR_HALF_WORLD
            - tile_y * TILE_SIZE * resolution
    )

    return min_x, max_y


def download_tile(
        session: requests.Session,
        zoom: int,
        tile_x: int,
        tile_y: int,
) -> tuple[np.ndarray, str]:
    url = TILE_URL.format(
        zoom=zoom,
        x=tile_x,
        y=tile_y,
    )

    last_error: Exception | None = None

    for attempt in range(1, 5):
        try:
            response = session.get(
                url,
                timeout=(15, 45),
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                "",
            )

            if (
                    content_type
                    and "image" not in content_type.lower()
            ):
                raise SampleDownloadError(
                    "Unexpected response type for "
                    f"{url}: {content_type}"
                )

            encoded = np.frombuffer(
                response.content,
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                encoded,
                cv2.IMREAD_COLOR,
            )

            if (
                    image is None
                    or image.shape[:2]
                    != (TILE_SIZE, TILE_SIZE)
            ):
                raise SampleDownloadError(
                    f"Invalid {TILE_SIZE}x{TILE_SIZE} "
                    f"tile returned for {url}"
                )

            return image, url

        except (
                requests.RequestException,
                SampleDownloadError,
        ) as error:
            last_error = error

            if attempt < 4:
                time.sleep(float(attempt))

    raise SampleDownloadError(
        "Could not download orthophoto tile "
        f"{url}: {last_error}"
    )


def site_paths(
        output_dir: Path,
        site: ViennaSite,
) -> tuple[Path, Path, Path]:
    base_name = f"vienna_{site.name}"

    return (
        output_dir / f"{base_name}.tif",
        output_dir
        / f"{base_name}_buildings.geojson",
        output_dir
        / f"{base_name}_metadata.json",
    )


def remove_incomplete_existing_set(
        output_dir: Path,
        site: ViennaSite,
) -> None:
    """Remove orphan files left by an older failed downloader."""

    paths = site_paths(
        output_dir=output_dir,
        site=site,
    )

    existing = [
        path.exists()
        for path in paths
    ]

    if any(existing) and not all(existing):
        print(
            f"{site.name}: removing an incomplete "
            "existing sample set."
        )

        for path in paths:
            path.unlink(missing_ok=True)


def build_mosaic(
        site: ViennaSite,
        zoom: int,
        tile_count: int,
) -> tuple[np.ndarray, list[str], int, int]:
    center_x, center_y = lonlat_to_tile(
        longitude=site.longitude,
        latitude=site.latitude,
        zoom=zoom,
    )

    half = tile_count // 2
    first_x = center_x - half
    first_y = center_y - half

    mosaic = np.zeros(
        (
            tile_count * TILE_SIZE,
            tile_count * TILE_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    urls: list[str] = []
    total_tiles = tile_count * tile_count

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "roof-analysis-propx-assessment/1.0 "
                    "(real open Vienna sample downloader)"
                )
            }
        )

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

                mosaic[
                    y0:y0 + TILE_SIZE,
                    x0:x0 + TILE_SIZE,
                ] = tile

                urls.append(url)

                completed = (
                        row * tile_count
                        + column
                        + 1
                )

                print(
                    f"{site.name}: downloaded tile "
                    f"{completed}/{total_tiles}",
                    end="\r",
                    flush=True,
                )

    print()

    return (
        mosaic,
        urls,
        first_x,
        first_y,
    )


def write_geotiff(
        path: Path,
        site: ViennaSite,
        mosaic: np.ndarray,
        first_x: int,
        first_y: int,
        zoom: int,
) -> None:
    min_x, max_y = tile_top_left_meters(
        tile_x=first_x,
        tile_y=first_y,
        zoom=zoom,
    )

    resolution = tile_resolution(zoom)

    transform = from_origin(
        min_x,
        max_y,
        resolution,
        resolution,
    )

    rgb = cv2.cvtColor(
        mosaic,
        cv2.COLOR_BGR2RGB,
    )

    with rasterio.open(
            path,
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
        dataset.write(
            np.moveaxis(
                rgb,
                -1,
                0,
            )
        )

        dataset.update_tags(
            SOURCE_DESCRIPTION=(
                "Real basemap.at orthophoto WMTS mosaic; "
                "Datenquelle: basemap.at, CC BY 4.0."
            ),
            SAMPLE_SITE=site.name,
            SAMPLE_DESCRIPTION=site.description,
        )


def fetch_footprints_with_retry(
        geotiff_path: Path,
        footprints_path: Path,
) -> int:
    geo_image = GeoTiffImageSource().load(
        geotiff_path
    )

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            fetcher = (
                OverpassBuildingFootprintFetcher(
                    timeout_seconds=180
                )
            )

            fetcher.fetch(
                geo_image=geo_image,
                output_path=footprints_path,
            )

            collection = (
                GeoJsonBuildingFootprintSource()
                .load(
                    geo_image=geo_image,
                    footprints_path=footprints_path,
                )
            )

            return len(collection.candidates)

        except (
                FootprintError,
                OSError,
                ValueError,
        ) as error:
            last_error = error

            footprints_path.unlink(
                missing_ok=True
            )

            if attempt < 3:
                wait_seconds = 3 * attempt

                print(
                    "Footprint download or validation "
                    "failed; retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    raise SampleDownloadError(
        "Could not download and validate the "
        "matching OpenStreetMap building "
        f"footprints: {last_error}"
    )


def write_metadata(
        path: Path,
        site: ViennaSite,
        zoom: int,
        tile_count: int,
        building_count: int,
        urls: list[str],
) -> None:
    metadata: dict[str, Any] = {
        "site": site.name,
        "description": site.description,
        "center_wgs84": [
            site.longitude,
            site.latitude,
        ],
        "zoom": zoom,
        "tile_count_per_side": tile_count,
        "source": (
            "basemap.at Orthofoto WMTS"
        ),
        "attribution": (
            "Datenquelle: basemap.at, CC BY 4.0"
        ),
        "building_source": (
            "OpenStreetMap contributors "
            "via Overpass API"
        ),
        "usable_independent_buildings": (
            building_count
        ),
        "tile_urls": urls,
    }

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def validate_staged_files(
        geotiff_path: Path,
        footprints_path: Path,
        metadata_path: Path,
        minimum_buildings: int,
) -> int:
    for path in (
            geotiff_path,
            footprints_path,
            metadata_path,
    ):
        if (
                not path.is_file()
                or path.stat().st_size == 0
        ):
            raise SampleDownloadError(
                "Required staged file is "
                f"missing or empty: {path}"
            )

    try:
        GeoTiffImageSource.validate(
            geotiff_path
        )

        geo_image = (
            GeoTiffImageSource()
            .load(geotiff_path)
        )

        collection = (
            GeoJsonBuildingFootprintSource()
            .load(
                geo_image=geo_image,
                footprints_path=footprints_path,
            )
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

    except (
            GeoTiffValidationError,
            FootprintError,
            OSError,
            ValueError,
            json.JSONDecodeError,
    ) as error:
        raise SampleDownloadError(
            "The staged sample failed "
            f"validation: {error}"
        ) from error

    building_count = len(
        collection.candidates
    )

    if building_count < minimum_buildings:
        raise SampleDownloadError(
            "The site produced only "
            f"{building_count} usable independent "
            f"buildings; {minimum_buildings} are "
            "required. Increase --tiles or use "
            "another site."
        )

    recorded_count = metadata.get(
        "usable_independent_buildings"
    )

    if recorded_count != building_count:
        raise SampleDownloadError(
            "Metadata building count does not "
            "match the validated GeoJSON."
        )

    return building_count


def publish_complete_set(
        staged_paths: tuple[Path, Path, Path],
        final_paths: tuple[Path, Path, Path],
) -> None:
    """Publish all files and roll back if replacement fails."""

    output_dir = final_paths[0].parent

    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=".vienna_sample_backup_",
            dir=output_dir,
        )
    )

    backups: dict[Path, Path] = {}
    published: list[Path] = []

    try:
        for final_path in final_paths:
            if final_path.exists():
                backup_path = (
                        backup_dir
                        / final_path.name
                )

                os.replace(
                    final_path,
                    backup_path,
                )

                backups[final_path] = (
                    backup_path
                )

        for staged_path, final_path in zip(
                staged_paths,
                final_paths,
                strict=True,
        ):
            os.replace(
                staged_path,
                final_path,
            )

            published.append(
                final_path
            )

        if not all(
                path.is_file()
                for path in final_paths
        ):
            raise SampleDownloadError(
                "The final sample set was not "
                "published completely."
            )

    except Exception:
        for final_path in published:
            final_path.unlink(
                missing_ok=True
            )

        for (
                final_path,
                backup_path,
        ) in backups.items():
            if backup_path.exists():
                os.replace(
                    backup_path,
                    final_path,
                )

        raise

    finally:
        shutil.rmtree(
            backup_dir,
            ignore_errors=True,
        )


def download_site(
        site: ViennaSite,
        output_dir: Path,
        zoom: int,
        tile_count: int,
        minimum_buildings: int,
) -> tuple[Path, Path, Path]:
    validate_options(
        zoom=zoom,
        tile_count=tile_count,
        minimum_buildings=minimum_buildings,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    remove_incomplete_existing_set(
        output_dir=output_dir,
        site=site,
    )

    final_paths = site_paths(
        output_dir=output_dir,
        site=site,
    )

    with tempfile.TemporaryDirectory(
            prefix=(
                    f".vienna_{site.name}_staging_"
            ),
            dir=output_dir,
    ) as temporary_directory:
        staging_dir = Path(
            temporary_directory
        )

        staged_paths = site_paths(
            output_dir=staging_dir,
            site=site,
        )

        (
            staged_geotiff,
            staged_footprints,
            staged_metadata,
        ) = staged_paths

        (
            mosaic,
            urls,
            first_x,
            first_y,
        ) = build_mosaic(
            site=site,
            zoom=zoom,
            tile_count=tile_count,
        )

        write_geotiff(
            path=staged_geotiff,
            site=site,
            mosaic=mosaic,
            first_x=first_x,
            first_y=first_y,
            zoom=zoom,
        )

        # Reject a broken raster before requesting footprints.
        GeoTiffImageSource.validate(
            staged_geotiff
        )

        building_count = (
            fetch_footprints_with_retry(
                geotiff_path=staged_geotiff,
                footprints_path=staged_footprints,
            )
        )

        if building_count < minimum_buildings:
            raise SampleDownloadError(
                f"Site {site.name} produced only "
                f"{building_count} usable independent "
                f"buildings; {minimum_buildings} are "
                "required. Increase --tiles or choose "
                "another site."
            )

        write_metadata(
            path=staged_metadata,
            site=site,
            zoom=zoom,
            tile_count=tile_count,
            building_count=building_count,
            urls=urls,
        )

        validate_staged_files(
            geotiff_path=staged_geotiff,
            footprints_path=staged_footprints,
            metadata_path=staged_metadata,
            minimum_buildings=minimum_buildings,
        )

        publish_complete_set(
            staged_paths=staged_paths,
            final_paths=final_paths,
        )

    return final_paths


def main() -> None:
    args = parse_args()

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    selected_sites = (
        list(SITES.values())
        if args.site == "all"
        else [SITES[args.site]]
    )

    try:
        for site in selected_sites:
            print(
                "\nDownloading complete sample: "
                f"{site.name}"
            )

            (
                geotiff_path,
                footprints_path,
                metadata_path,
            ) = download_site(
                site=site,
                output_dir=output_dir,
                zoom=args.zoom,
                tile_count=args.tiles,
                minimum_buildings=(
                    args.minimum_buildings
                ),
            )

            print(
                f"Created: {geotiff_path}"
            )

            print(
                f"Created: {footprints_path}"
            )

            print(
                f"Created: {metadata_path}"
            )

    except (
            SampleDownloadError,
            GeoTiffValidationError,
            FootprintError,
            OSError,
            requests.RequestException,
            ValueError,
    ) as error:
        raise SystemExit(
            f"Error: {error}"
        ) from error


if __name__ == "__main__":
    main()