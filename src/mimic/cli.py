"""Command-line interface for one-shot MIMIC data generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mimic import mimic_data


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.filename)
    if not input_path.exists():
        parser.error(f"input file does not exist: {input_path}")

    columns = _parse_columns(args.columns, parser)
    try:
        df = _read_frame(input_path)
    except ValueError as exc:
        parser.error(str(exc))
    synthetic = mimic_data(
        df,
        columns=columns,
        mode=args.mode,
        capacity=args.capacity,
        random_state=args.random_state,
    )

    output_path = Path(args.output) if args.output is not None else _default_output_path(input_path)
    try:
        _write_frame(synthetic, output_path)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {len(synthetic)} rows to {output_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mimic-data",
        description=(
            "Fit MIMIC on a tabular data file and write a synthetic dataset with "
            "the same number of rows. CSV, Parquet, and Excel files are supported."
        ),
    )
    parser.add_argument("filename", help="Input .csv, .parquet, .xls, or .xlsx file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename. Defaults to '<input_stem>_mimic<input_suffix>' in the same directory.",
    )
    parser.add_argument(
        "--columns",
        default="auto",
        help=(
            "Column role specification. Use 'auto' for heuristic inference, or pass a JSON mapping "
            "such as '{\"ignore\":[\"id\"],\"regression\":[\"age\"],\"classification\":[\"segment\"]}'. "
            "Prefix with @ to read the JSON mapping from a file."
        ),
    )
    parser.add_argument(
        "--mode",
        default="factorised",
        choices=["identity", "direct", "factorised", "joint", "0", "1", "2", "3"],
        help="MIMIC preset mode. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--capacity",
        type=float,
        default=0.25,
        help="Model capacity in [0, 1]. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Optional random seed for reproducible generation.",
    )
    return parser


def _parse_columns(value: str, parser: argparse.ArgumentParser):
    if value in {"auto", "None", "none"}:
        return "auto" if value == "auto" else None
    try:
        if value.startswith("@"):
            text = Path(value[1:]).read_text()
        else:
            text = value
        parsed = json.loads(text)
    except OSError as exc:
        parser.error(f"could not read columns file {value!r}: {exc}")
    except json.JSONDecodeError as exc:
        parser.error(f"--columns must be 'auto' or valid JSON: {exc}")
    if not isinstance(parsed, dict):
        parser.error("--columns JSON must be an object mapping roles to column lists")
    return parsed


def _read_frame(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path)
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path)
    if suffix in {".xls", ".xlsx"}:
        import pandas as pd

        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format {suffix!r}; expected .csv, .parquet, .xls, or .xlsx")


def _write_frame(df, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix in {".xls", ".xlsx"}:
        df.to_excel(path, index=False)
        return
    raise ValueError(f"Unsupported output format {suffix!r}; expected .csv, .parquet, .xls, or .xlsx")


def _default_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_mimic{path.suffix}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
