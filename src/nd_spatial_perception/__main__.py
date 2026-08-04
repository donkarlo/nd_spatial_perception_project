from __future__ import annotations

import argparse
from pathlib import Path

from building.kind.roof_analysis.application import RoofAnalysisApplication
from building.kind.roof_analysis.attributes import RoofAttributeExtractor
from building.kind.roof_analysis.detector import MultiMaterialRoofDetector
from building.kind.roof_analysis.exporter import ResultExporter
from building.kind.roof_analysis.image_source import GeoTiffImageSource

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect roofs in one georeferenced aerial GeoTIFF and export "
            "roof attributes as JSON."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Path to a georeferenced GeoTIFF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--max-buildings",
        type=int,
        default=None,
        help="Maximum number of detected buildings",
    )
    parser.add_argument(
        "--min-area-px",
        type=float,
        default=None,
        help="Minimum candidate area in pixels",
    )

    return parser.parse_args()


def ask_path(message: str, default: Path | None = None) -> Path:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        value = input(f"{message}{default_text}: ").strip()

        if not value and default is not None:
            return default

        if value:
            return Path(value).expanduser()

        print("A value is required.")


def ask_int(message: str, default: int, minimum: int) -> int:
    while True:
        value = input(f"{message} [{default}]: ").strip()

        if not value:
            return default

        try:
            number = int(value)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if number < minimum:
            print(f"The value must be at least {minimum}.")
            continue

        return number


def ask_float(message: str, default: float, minimum: float) -> float:
    while True:
        value = input(f"{message} [{default}]: ").strip()

        if not value:
            return default

        try:
            number = float(value)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if number <= minimum:
            print(f"The value must be greater than {minimum}.")
            continue

        return number


def complete_missing_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.image is None:
        args.image = ask_path(
            "GeoTIFF image path",
        )

    if args.output is None:
        args.output = ask_path(
            "Output directory",
            default=PROJECT_ROOT / "outputs",
        )

    if args.max_buildings is None:
        args.max_buildings = ask_int(
            "Maximum number of buildings",
            default=10,
            minimum=1,
        )

    if args.min_area_px is None:
        args.min_area_px = ask_float(
            "Minimum roof area in pixels",
            default=450.0,
            minimum=0.0,
        )

    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.exists():
        raise SystemExit(f"Image file does not exist: {args.image}")

    if not args.image.is_file():
        raise SystemExit(f"Image path is not a file: {args.image}")

    if args.image.suffix.lower() not in {".tif", ".tiff"}:
        raise SystemExit("The input image must be a GeoTIFF file (.tif or .tiff).")

    if args.max_buildings < 1:
        raise SystemExit("--max-buildings must be at least 1")

    if args.min_area_px <= 0:
        raise SystemExit("--min-area-px must be positive")


def main() -> None:
    args = complete_missing_args(parse_args())

    if not args.output.is_absolute():
        args.output = PROJECT_ROOT / args.output

    validate_args(args)

    application = RoofAnalysisApplication(
        image_source=GeoTiffImageSource(),
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


if __name__ == "__main__":
    sample_path = (
        "/home/donkarlo/Dropbox/repo/nd_spatial_perception_project/data/sample_vienna.tif"
    )
    main()