from __future__ import annotations

import argparse
from pathlib import Path

from kind.building.kind.roof_analysis.application import RoofAnalysisApplication
from kind.building.kind.roof_analysis.attributes import RoofAttributeExtractor
from kind.building.kind.roof_analysis.detector import MultiMaterialRoofDetector
from kind.building.kind.roof_analysis.exporter import ResultExporter
from kind.building.kind.roof_analysis.image_source import (
    GeoTiffImageSource,
    GeoTiffValidationError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IMAGE = PROJECT_ROOT / "data" / "sample_vienna.tif"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs"
DEFAULT_MAX_BUILDINGS = 10
DEFAULT_MIN_AREA_PX = 450.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect roofs in one georeferenced aerial GeoTIFF and export "
            "roof attributes as JSON. Run without arguments for interactive mode."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Path to a georeferenced RGB GeoTIFF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
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


def resolve_path(path: Path) -> Path:
    path = path.expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def ask_geotiff_path(
        message: str,
        default: Path | None = None,
) -> Path:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        value = input(f"{message}{default_text}: ").strip()

        if not value and default is not None:
            selected_path = default
        elif value:
            selected_path = Path(value)
        else:
            print("Error: an input path is required.")
            continue

        selected_path = resolve_path(selected_path)

        try:
            GeoTiffImageSource.validate(selected_path)
        except GeoTiffValidationError as error:
            print()
            print(f"Error: {error}")
            print("Please select a valid georeferenced RGB GeoTIFF.")
            print()
            continue

        return selected_path


def ask_path(message: str, default: Path) -> Path:
    value = input(f"{message} [{default}]: ").strip()
    selected_path = Path(value) if value else default
    return resolve_path(selected_path)


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


def validate_numeric_args(args: argparse.Namespace) -> None:
    if args.max_buildings < 1:
        raise ValueError("--max-buildings must be at least 1")

    if args.min_area_px <= 0:
        raise ValueError("--min-area-px must be positive")


def create_interactive_args() -> argparse.Namespace:
    image = ask_geotiff_path(
        "GeoTIFF image path",
        default=DEFAULT_IMAGE,
    )

    return argparse.Namespace(
        image=image,
        output=ask_path(
            "Output directory",
            default=DEFAULT_OUTPUT,
        ),
        max_buildings=ask_int(
            "Maximum number of buildings",
            default=DEFAULT_MAX_BUILDINGS,
            minimum=1,
        ),
        min_area_px=ask_float(
            "Minimum roof area in pixels",
            default=DEFAULT_MIN_AREA_PX,
            minimum=0.0,
        ),
    )


def run_analysis(args: argparse.Namespace) -> None:
    args.image = resolve_path(args.image)
    args.output = resolve_path(args.output)

    validate_numeric_args(args)

    image_source = GeoTiffImageSource()
    image_source.validate(args.image)

    application = RoofAnalysisApplication(
        image_source=image_source,
        detector=MultiMaterialRoofDetector(
            min_area_px=args.min_area_px,
        ),
        extractor=RoofAttributeExtractor(),
        exporter=ResultExporter(),
    )

    count = application.run(
        image_path=args.image,
        output_dir=args.output,
        max_buildings=args.max_buildings,
    )

    print()
    print(f"Detected buildings: {count}")
    print(f"JSON: {args.output / 'roof_attributes.json'}")
    print(f"Overlays: {args.output / 'overlays'}")


def run_interactive() -> None:
    print("Interactive roof analysis")
    print("Press Enter to accept each value shown in brackets.")

    while True:
        print()
        args = create_interactive_args()

        try:
            run_analysis(args)
        except (GeoTiffValidationError, OSError, RuntimeError, ValueError) as error:
            print()
            print(f"Analysis failed: {error}")

        print()

        if not ask_yes_no("Do you want to analyze another image?"):
            print("Finished.")
            return


def main() -> None:
    args = parse_args()

    if args.image is None:
        try:
            run_interactive()
        except (EOFError, KeyboardInterrupt):
            print("\nFinished.")
        return

    try:
        run_analysis(args)
    except (GeoTiffValidationError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()