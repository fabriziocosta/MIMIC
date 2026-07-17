"""Configuration helpers for the streamlined manuscript experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
import ast
import re


METHOD_KEYS = (
    "real_balanced",
    "direct_smote",
    "direct_displacement",
    "latent_smote",
    "latent_displacement",
)

ARTIFACT_FILENAMES = {
    "raw_results": Path("raw") / "condition_results.csv",
    "learning_curves": Path("tables") / "learning_curves.csv",
    "aulc": Path("tables") / "aulc.csv",
    "pairwise": Path("tables") / "pairwise_comparisons.csv",
    "real_equivalence": Path("tables") / "real_equivalent_sample_fraction.csv",
    "real_equivalence_summary": Path("tables") / "real_equivalent_sample_fraction_summary.csv",
    "regime": Path("tables") / "regime_summary.csv",
    "rank": Path("tables") / "rank_summary.csv",
    "conclusions": Path("reports") / "prescriptive_conclusions.md",
    "embedding_models": Path("models") / "latent_displacement",
}

_EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    datasets: tuple[str, ...]
    imbalance_ratios: tuple[float, ...]
    training_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    methods: tuple[str, ...] = METHOD_KEYS
    dataset_n_rows: dict[str, int | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    profile: ProfileConfig
    artifact_dir: Path = Path("manuscript/experiments/artifacts")
    test_size: float = 0.3
    target_column: str = "label"
    minority_label: str = "minority"
    majority_label: str = "majority"
    n_neighbors: int = 5
    lambda_range: tuple[float, float] = (0.0, 1.0)
    mimic_mode: str = "factorised"
    mimic_capacity: float = 0.25
    repair_direct_samples: bool = True
    equivalence_margin: float = 0.01
    aulc_segments: dict[str, tuple[int | None, int | None]] = field(default_factory=dict)

    @property
    def artifact_root(self) -> Path:
        return Path(self.artifact_dir)


def load_config(path: str | Path, *, artifact_dir: str | Path | None = None) -> ExperimentConfig:
    data = _load_simple_yaml(Path(path))
    profile_data = data["profile"]
    profile = ProfileConfig(
        name=str(profile_data["name"]),
        datasets=tuple(str(v) for v in profile_data["datasets"]),
        imbalance_ratios=tuple(_parse_ratio(v) for v in profile_data["imbalance_ratios"]),
        training_sizes=tuple(int(v) for v in profile_data["training_sizes"]),
        seeds=tuple(int(v) for v in profile_data["seeds"]),
        methods=tuple(str(v) for v in profile_data.get("methods", METHOD_KEYS)),
        dataset_n_rows={str(k): (None if v is None else int(v)) for k, v in profile_data.get("dataset_n_rows", {}).items()},
    )
    unknown_methods = set(profile.methods) - set(METHOD_KEYS)
    if unknown_methods:
        raise ValueError(f"Unknown methods in config: {sorted(unknown_methods)}")
    selected_artifact_dir = artifact_dir if artifact_dir is not None else data.get("artifact_dir", "manuscript/experiments/artifacts")
    return ExperimentConfig(
        profile=profile,
        artifact_dir=Path(selected_artifact_dir),
        test_size=float(data.get("test_size", 0.3)),
        n_neighbors=int(data.get("n_neighbors", 5)),
        lambda_range=tuple(float(v) for v in data.get("lambda_range", [0.0, 1.0])),
        mimic_mode=str(data.get("mimic_mode", "factorised")),
        mimic_capacity=float(data.get("mimic_capacity", 0.25)),
        repair_direct_samples=bool(data.get("repair_direct_samples", True)),
        equivalence_margin=float(data.get("equivalence_margin", 0.01)),
        aulc_segments={
            str(k): _parse_segment(v)
            for k, v in data.get("aulc_segments", {}).items()
        },
    )


def artifact_paths(config: ExperimentConfig) -> dict[str, Path]:
    paths = {key: config.artifact_root / rel for key, rel in ARTIFACT_FILENAMES.items()}
    paths["figures"] = config.artifact_root / "figures"
    return paths


def experiment_artifact_dir(
    artifact_base_dir: str | Path,
    experiment_name: str,
) -> Path:
    """Return an isolated artifact directory for a safely named experiment."""
    name = experiment_name.strip()
    if not _EXPERIMENT_NAME_PATTERN.fullmatch(name) or name in {".", ".."}:
        raise ValueError(
            "experiment_name must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    return Path(artifact_base_dir) / name


def resolve_experiment_artifact_dir(
    artifact_base_dir: str | Path,
    experiment_name: str | None,
) -> Path:
    """Resolve a named experiment, or the most recently updated completed run."""
    if experiment_name is not None:
        return experiment_artifact_dir(artifact_base_dir, experiment_name)

    base_dir = Path(artifact_base_dir)
    result_files = [
        path
        for path in base_dir.glob("*/raw/condition_results.csv")
        if path.is_file()
    ]
    if not result_files:
        raise FileNotFoundError(
            f"No experiment results found under artifact directory: {base_dir}"
        )
    latest_results = max(result_files, key=lambda path: path.stat().st_mtime_ns)
    return latest_results.parent.parent


def with_dataset_n_rows(
    config: ExperimentConfig,
    n_rows: int,
) -> ExperimentConfig:
    """Apply one explicit dataset row limit to every dataset in a profile."""
    selected_n_rows = int(n_rows)
    if selected_n_rows < 1:
        raise ValueError("n_rows must be at least 1")
    profile = dataclass_replace(
        config.profile,
        dataset_n_rows={key: selected_n_rows for key in config.profile.datasets},
    )
    return dataclass_replace(config, profile=profile)


def ensure_artifact_dirs(config: ExperimentConfig) -> None:
    for path in artifact_paths(config).values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _parse_ratio(value) -> float:
    if isinstance(value, str) and ":" in value:
        left, right = value.split(":", 1)
        return float(left) / float(right)
    return float(value)


def _parse_segment(value) -> tuple[int | None, int | None]:
    if isinstance(value, dict):
        start = value.get("min")
        end = value.get("max")
        return (None if start is None else int(start), None if end is None else int(end))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (None if value[0] is None else int(value[0]), None if value[1] is None else int(value[1]))
    raise ValueError(f"Invalid AULC segment: {value!r}")


def _load_simple_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _parse_simple_yaml(path.read_text())
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}")
    return loaded


def _parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if raw_value == "":
            child: dict = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(raw_value)
    return root


def _parse_scalar(raw: str):
    if raw in {"null", "None", "~"}:
        return None
    if raw in {"true", "True"}:
        return True
    if raw in {"false", "False"}:
        return False
    if raw.startswith("[") or raw.startswith("{"):
        return ast.literal_eval(raw.replace("null", "None").replace("true", "True").replace("false", "False"))
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return ast.literal_eval(raw)
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw
