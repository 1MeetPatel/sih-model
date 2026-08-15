#!/usr/bin/env python3
"""
Production Satellite Road Extraction & Temporal Change Detection System
========================================================================
Main Command-Line Entry Point
"""

import sys
import argparse
from src.cli.commands import inspect_command
from src.utils.logging import get_logger

logger = get_logger("Main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Production Geospatial AI Road Extraction & Change Detection Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available sub-commands")

    # Command: inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect GeoTIFF raster metadata, CRS, GSD, and resolution feasibility")
    p_inspect.add_argument("--input", "-i", required=True, help="Path to input GeoTIFF image")
    p_inspect.add_argument("--verbose", "-v", action="store_true", help="Enable verbose diagnostic logs")

    # Command: extract
    p_extract = subparsers.add_parser("extract", help="Extract roads >= 6.1m from satellite GeoTIFF")
    p_extract.add_argument("--input", "-i", required=True, help="Path to input satellite GeoTIFF")
    p_extract.add_argument("--checkpoint", "-c", required=True, help="Path to pre-trained PyTorch model checkpoint (.pth)")
    p_extract.add_argument("--output", "-o", default="outputs/", help="Directory to store extracted outputs (GPKG, GeoJSON, TIFFs)")
    p_extract.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file")
    p_extract.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Compute device for neural inference")
    p_extract.add_argument("--min-width", type=float, default=6.096, help="Minimum qualifying road width in meters (default: 6.096m / 20ft)")
    p_extract.add_argument("--threshold", type=float, default=0.35, help="Probability threshold for initial road segmentation")

    # Command: change
    p_change = subparsers.add_parser("change", help="Perform temporal change detection between Before and After GeoTIFFs")
    p_change.add_argument("--before", "-b", required=True, help="Path to BEFORE satellite GeoTIFF")
    p_change.add_argument("--after", "-a", required=True, help="Path to AFTER satellite GeoTIFF")
    p_change.add_argument("--checkpoint", "-c", required=True, help="Path to pre-trained PyTorch model checkpoint (.pth)")
    p_change.add_argument("--output", "-o", default="outputs/changes/", help="Directory to store change detection GIS outputs")
    p_change.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file")
    p_change.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    p_change.add_argument("--min-change-length", type=float, default=20.0, help="Minimum change segment length in meters (default: 20m)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        inspect_command(input_path=args.input, verbose=args.verbose)
    elif args.command == "extract":
        from src.cli.commands import extract_command
        extract_command(args)
    elif args.command == "change":
        from src.cli.commands import change_command
        change_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
