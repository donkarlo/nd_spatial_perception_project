from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from kind.building.kind.roof_analysis.application import RoofAnalysisApplication
from kind.building.kind.roof_analysis.attributes import RoofAttributeExtractor
from kind.building.kind.roof_analysis.detector import MultiMaterialRoofDetector
from kind.building.kind.roof_analysis.exporter import ResultExporter
from kind.building.kind.roof_analysis.image_source import (
    GeoTiffImageSource,
    GeoTiffValidationError,
)

DEFAULT_MAX_BUILDINGS = 10
DEFAULT_MIN_AREA_PX = 450.0


def project_root() -> Path:
    """Return the directory from which the application was started."""
    return Path.cwd().resolve()


def default_image_path() -> Path:
    return project_root() / "data" / "sample_vienna.tif"


def default_output_root() -> Path:
    return project_root() / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect roofs in a georeferenced aerial GeoTIFF and save each "
            "run in outputs/<image-name>."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Absolute path to a georeferenced RGB GeoTIFF",
    )
    parser.add_argument(
        "--output",
        "--output-root",
        dest="output_root",
        type=Path,
        default=None,
        help="Root directory for image-specific result directories",
    )
    parser.add_argument(
        "--max-buildings",
        type=int,
        default=DEFAULT_MAX_BUILDINGS,
        help=(
            "Maximum number of detected buildings "
            f"(default: {DEFAULT_MAX_BUILDINGS})"
        ),
    )
    parser.add_argument(
        "--min-area-px",
        type=float,
        default=DEFAULT_MIN_AREA_PX,
        help=(
            "Minimum candidate area in pixels "
            f"(default: {DEFAULT_MIN_AREA_PX:g})"
        ),
    )

    return parser.parse_args()


def require_absolute_path(path: Path, label: str) -> Path:
    resolved_path = path.expanduser()

    if not resolved_path.is_absolute():
        raise ValueError(
            f"{label} must be an absolute path.\n"
            f"Received: {path}"
        )

    return resolved_path.resolve(strict=False)


def ask_geotiff_path() -> Path:
    default = default_image_path()

    while True:
        value = input(f"Absolute GeoTIFF image path [{default}]: ").strip()
        selected = Path(value) if value else default

        try:
            selected = require_absolute_path(selected, "The image path")
            GeoTiffImageSource.validate(selected)
        except (GeoTiffValidationError, ValueError) as error:
            print()
            print(f"Error: {error}")
            print("Please select a valid georeferenced RGB GeoTIFF.")
            print()
            continue

        return selected


def ask_output_root() -> Path:
    default = default_output_root()

    while True:
        value = input(f"Absolute output root directory [{default}]: ").strip()
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


def validate_options(max_buildings: int, min_area_px: float) -> None:
    if max_buildings < 1:
        raise ValueError("--max-buildings must be at least 1")

    if min_area_px <= 0:
        raise ValueError("--min-area-px must be positive")


def prepare_output_directory(
    output_root: Path,
    image_path: Path,
) -> Path:
    """Create a clean output directory for the current input filename."""
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
    output_root: Path,
    max_buildings: int,
    min_area_px: float,
) -> None:
    image_path = require_absolute_path(image_path, "The image path")
    output_root = require_absolute_path(output_root, "The output directory")

    validate_options(max_buildings, min_area_px)
    GeoTiffImageSource.validate(image_path)

    output_dir = prepare_output_directory(output_root, image_path)

    application = RoofAnalysisApplication(
        image_source=GeoTiffImageSource(),
        detector=MultiMaterialRoofDetector(min_area_px=min_area_px),
        extractor=RoofAttributeExtractor(),
        exporter=ResultExporter(),
    )

    count = application.run(
        image_path=image_path,
        output_dir=output_dir,
        max_buildings=max_buildings,
    )

    print()
    print(f"Detected buildings: {count}")
    print(f"Results directory: {output_dir}")
    print(f"JSON: {output_dir / 'roof_attributes.json'}")
    print(f"Overlays: {output_dir / 'overlays'}")


def run_interactive() -> None:
    print("Interactive roof analysis")
    print("Press Enter to accept each value shown in brackets.")

    while True:
        print()
        image_path = ask_geotiff_path()
        output_root = ask_output_root()
        max_buildings = ask_int(
            "Maximum number of buildings",
            DEFAULT_MAX_BUILDINGS,
            minimum=1,
        )
        min_area_px = ask_float(
            "Minimum roof area in pixels",
            DEFAULT_MIN_AREA_PX,
            minimum=0.0,
        )

        try:
            run_analysis(
                image_path=image_path,
                output_root=output_root,
                max_buildings=max_buildings,
                min_area_px=min_area_px,
            )
        except (GeoTiffValidationError, OSError, RuntimeError, ValueError) as error:
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
        output_root=output_root,
        max_buildings=args.max_buildings,
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
    except (GeoTiffValidationError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
