from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .application import RoofAnalysisApplication
from .attributes import RoofAttributeExtractor
from .exporter import ResultExporter
from .footprints import (
    FootprintError,
    GeoJsonBuildingFootprintSource,
    OverpassBuildingFootprintFetcher,
)
from .image_source import GeoTiffImageSource, GeoTiffValidationError

DEFAULT_MAX_BUILDINGS = 10
DEFAULT_MIN_VISIBLE_BUILDINGS = 20
DEFAULT_MIN_AREA_M2 = 20.0
DEFAULT_MIN_AREA_PX = 300.0


def project_root() -> Path:
    return Path.cwd().resolve()


def default_output_root() -> Path:
    return project_root() / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse a georeferenced aerial GeoTIFF with open building "
            "footprints, require at least 20 independent buildings, and "
            "extract attributes for 10 selected roofs."
        )
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Absolute path to a georeferenced RGB GeoTIFF",
    )
    parser.add_argument(
        "--footprints",
        type=Path,
        default=None,
        help="Absolute path to a building-footprint GeoJSON file",
    )
    parser.add_argument(
        "--fetch-osm",
        action="store_true",
        help="Download OSM building footprints for the GeoTIFF extent",
    )
    parser.add_argument(
        "--output-root",
        "--output",
        dest="output_root",
        type=Path,
        default=None,
        help="Root directory for image-specific output directories",
    )
    parser.add_argument(
        "--max-buildings",
        type=int,
        default=DEFAULT_MAX_BUILDINGS,
        help=f"Number of roofs to export (default: {DEFAULT_MAX_BUILDINGS})",
    )
    parser.add_argument(
        "--min-visible-buildings",
        type=int,
        default=DEFAULT_MIN_VISIBLE_BUILDINGS,
        help=(
            "Minimum independent buildings required in the image "
            f"(default: {DEFAULT_MIN_VISIBLE_BUILDINGS})"
        ),
    )
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=DEFAULT_MIN_AREA_M2,
        help=(
            "Minimum footprint area in square metres "
            f"(default: {DEFAULT_MIN_AREA_M2:g})"
        ),
    )
    parser.add_argument(
        "--min-area-px",
        type=float,
        default=DEFAULT_MIN_AREA_PX,
        help=(
            "Minimum footprint area in pixels "
            f"(default: {DEFAULT_MIN_AREA_PX:g})"
        ),
    )
    return parser.parse_args()


def require_absolute_path(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"{label} must be an absolute path.\nReceived: {path}"
        )
    return expanded.resolve(strict=False)


def ask_geotiff_path(default: Path | None = None) -> Path:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"Absolute GeoTIFF image path{suffix}: ").strip()
        if not value and default is None:
            print("Error: an input GeoTIFF path is required.")
            continue
        selected = Path(value) if value else default
        assert selected is not None
        try:
            selected = require_absolute_path(selected, "The image path")
            GeoTiffImageSource.validate(selected)
            return selected
        except (GeoTiffValidationError, ValueError) as error:
            print()
            print(f"Error: {error}")
            print("Please select a valid georeferenced RGB GeoTIFF.")
            print()


def ask_footprint_choice() -> tuple[Path | None, bool]:
    while True:
        value = input(
            "Absolute building-footprints GeoJSON path "
            "[press Enter to fetch OSM automatically]: "
        ).strip()
        if not value:
            return None, True
        try:
            selected = require_absolute_path(
                Path(value),
                "The footprint path",
            )
            if not selected.exists() or not selected.is_file():
                raise ValueError(
                    f"The footprint file does not exist:\n{selected}"
                )
            if selected.suffix.lower() not in {".geojson", ".json"}:
                raise ValueError(
                    "The footprint file must end in .geojson or .json."
                )
            return selected, False
        except ValueError as error:
            print()
            print(f"Error: {error}")
            print()


def ask_output_root() -> Path:
    default = default_output_root()
    while True:
        value = input(
            f"Absolute output root directory [{default}]: "
        ).strip()
        selected = Path(value) if value else default
        try:
            return require_absolute_path(selected, "The output directory")
        except ValueError as error:
            print(f"Error: {error}")


def ask_int(message: str, default: int, minimum: int) -> int:
    while True:
        value = input(f"{message} [{default}]: ").strip()
        if not value:
            return default
        try:
            number = int(value)
        except ValueError:
            print("Error: please enter a valid integer.")
            continue
        if number < minimum:
            print(f"Error: the value must be at least {minimum}.")
            continue
        return number


def ask_float(message: str, default: float, minimum: float) -> float:
    while True:
        value = input(f"{message} [{default:g}]: ").strip()
        if not value:
            return default
        try:
            number = float(value)
        except ValueError:
            print("Error: please enter a valid number.")
            continue
        if number <= minimum:
            print(f"Error: the value must be greater than {minimum:g}.")
            continue
        return number


def ask_yes_no(message: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{message} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Error: please enter y or n.")


def validate_options(
    max_buildings: int,
    min_visible_buildings: int,
    min_area_m2: float,
    min_area_px: float,
) -> None:
    if max_buildings < 1:
        raise ValueError("--max-buildings must be at least 1")
    if min_visible_buildings < max_buildings:
        raise ValueError(
            "--min-visible-buildings cannot be smaller than --max-buildings"
        )
    if min_area_m2 <= 0.0:
        raise ValueError("--min-area-m2 must be positive")
    if min_area_px <= 0.0:
        raise ValueError("--min-area-px must be positive")


def prepare_output_directory(
    output_root: Path,
    image_path: Path,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / image_path.stem
    if output_dir.exists():
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run_analysis(
    image_path: Path,
    footprints_path: Path | None,
    fetch_osm: bool,
    output_root: Path,
    max_buildings: int,
    min_visible_buildings: int,
    min_area_m2: float,
    min_area_px: float,
) -> None:
    image_path = require_absolute_path(image_path, "The image path")
    output_root = require_absolute_path(output_root, "The output directory")
    if footprints_path is not None:
        footprints_path = require_absolute_path(
            footprints_path,
            "The footprint path",
        )

    validate_options(
        max_buildings=max_buildings,
        min_visible_buildings=min_visible_buildings,
        min_area_m2=min_area_m2,
        min_area_px=min_area_px,
    )
    GeoTiffImageSource.validate(image_path)
    output_dir = prepare_output_directory(output_root, image_path)

    application = RoofAnalysisApplication(
        image_source=GeoTiffImageSource(),
        footprint_source=GeoJsonBuildingFootprintSource(
            min_area_m2=min_area_m2,
            min_area_px=min_area_px,
        ),
        extractor=RoofAttributeExtractor(),
        exporter=ResultExporter(),
        overpass_fetcher=OverpassBuildingFootprintFetcher(),
    )
    try:
        visible_count, selected_count, resolved_footprints = application.run(
            image_path=image_path,
            output_dir=output_dir,
            footprints_path=footprints_path,
            fetch_osm=fetch_osm,
            max_buildings=max_buildings,
            min_visible_buildings=min_visible_buildings,
        )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    print()
    print(f"Usable independent buildings in image: {visible_count}")
    print(f"Selected roofs: {selected_count}")
    print(f"Footprints used: {resolved_footprints}")
    print(f"Results directory: {output_dir}")
    print(f"JSON: {output_dir / 'roof_attributes.json'}")
    print(f"Overlays: {output_dir / 'overlays'}")


def run_interactive() -> None:
    print("Interactive roof analysis")
    print("Press Enter to accept a displayed default value.")
    while True:
        print()
        image_path = ask_geotiff_path()
        footprints_path, fetch_osm = ask_footprint_choice()
        output_root = ask_output_root()
        max_buildings = ask_int(
            "Number of roofs to select",
            DEFAULT_MAX_BUILDINGS,
            minimum=1,
        )
        min_visible_buildings = ask_int(
            "Minimum independent buildings required in the image",
            DEFAULT_MIN_VISIBLE_BUILDINGS,
            minimum=max_buildings,
        )
        min_area_m2 = ask_float(
            "Minimum building area in square metres",
            DEFAULT_MIN_AREA_M2,
            minimum=0.0,
        )
        min_area_px = ask_float(
            "Minimum building area in pixels",
            DEFAULT_MIN_AREA_PX,
            minimum=0.0,
        )

        try:
            run_analysis(
                image_path=image_path,
                footprints_path=footprints_path,
                fetch_osm=fetch_osm,
                output_root=output_root,
                max_buildings=max_buildings,
                min_visible_buildings=min_visible_buildings,
                min_area_m2=min_area_m2,
                min_area_px=min_area_px,
            )
        except (
            FootprintError,
            GeoTiffValidationError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            print()
            print(f"Analysis failed: {error}")

        print()
        if not ask_yes_no("Do you want to analyze another image?"):
            print("Finished.")
            return


def run_non_interactive(args: argparse.Namespace) -> None:
    image_path = require_absolute_path(args.image, "The image path")
    output_root = (
        require_absolute_path(args.output_root, "The output directory")
        if args.output_root is not None
        else default_output_root()
    )
    run_analysis(
        image_path=image_path,
        footprints_path=args.footprints,
        fetch_osm=args.fetch_osm,
        output_root=output_root,
        max_buildings=args.max_buildings,
        min_visible_buildings=args.min_visible_buildings,
        min_area_m2=args.min_area_m2,
        min_area_px=args.min_area_px,
    )


def main() -> None:
    args = parse_args()
    try:
        if args.image is None:
            run_interactive()
        else:
            run_non_interactive(args)
    except (EOFError, KeyboardInterrupt):
        print("\nFinished.")
    except (
        FootprintError,
        GeoTiffValidationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
