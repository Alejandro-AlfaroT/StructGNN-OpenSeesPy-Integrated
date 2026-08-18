"""Dataset discovery, leakage-resistant splits, normalization, and batching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    sample_id: str
    case_id: str
    run_name: str
    record_id_x: str
    record_id_y: str

    @property
    def record_group(self) -> str:
        """Ground-motion pair used as an indivisible split group."""
        return f"{self.record_id_x}|{self.record_id_y}"


def _case_id_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("case_"):
            return parent.name
    return path.parent.parent.name


def discover_samples(dataset_roots: Iterable[str | Path]) -> list[SampleRecord]:
    """Discover successful compiled samples beneath one or more local roots."""
    records: list[SampleRecord] = []
    seen_ids: set[str] = set()
    for root_value in dataset_roots:
        root = Path(root_value).resolve()
        for sample_path in sorted(root.rglob("hybrid_sample.npz")):
            metadata_path = sample_path.with_name("hybrid_metadata.json")
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if bool(metadata.get("analysis_failed")):
                continue
            requested = int(metadata.get("npts_requested") or 0)
            completed = int(metadata.get("completed_steps") or 0)
            if requested <= 0 or requested != completed:
                continue
            case_id = _case_id_from_path(sample_path)
            run_name = str(metadata.get("run_name") or sample_path.parent.name)
            sample_id = f"{case_id}/{run_name}"
            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate sample identity {sample_id!r}; roots must contain "
                    "non-overlapping case ranges."
                )
            seen_ids.add(sample_id)
            records.append(
                SampleRecord(
                    path=sample_path,
                    sample_id=sample_id,
                    case_id=case_id,
                    run_name=run_name,
                    record_id_x=str(metadata.get("record_id_x") or ""),
                    record_id_y=str(metadata.get("record_id_y") or ""),
                )
            )
    return sorted(records, key=lambda item: item.sample_id)


def _group_targets(
    group_count: int,
    train_fraction: float,
    validation_fraction: float,
    test_group_count: int | None,
) -> dict[str, int]:
    train_groups = max(1, int(round(group_count * train_fraction)))
    if test_group_count is None:
        validation_groups = max(1, int(round(group_count * validation_fraction)))
        if train_groups + validation_groups >= group_count:
            validation_groups = 1
            train_groups = group_count - 2
        test_groups = group_count - train_groups - validation_groups
    else:
        test_groups = int(test_group_count)
        if not 1 <= test_groups <= group_count - 2:
            raise ValueError(
                "test_group_count must leave at least one train and one validation group."
            )
        if train_groups + test_groups >= group_count:
            train_groups = group_count - test_groups - 1
        validation_groups = group_count - train_groups - test_groups
    return {"train": train_groups, "validation": validation_groups, "test": test_groups}


def grouped_split(
    records: Sequence[SampleRecord],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 20260809,
    test_group_count: int | None = None,
) -> dict[str, list[SampleRecord]]:
    """Split by ground-motion pair so no earthquake pair leaks across splits."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than one.")

    groups: dict[str, list[SampleRecord]] = {}
    for record in records:
        groups.setdefault(record.record_group, []).append(record)
    group_names = sorted(groups)
    if len(group_names) < 3:
        raise ValueError("At least three distinct ground-motion groups are required.")
    random.Random(seed).shuffle(group_names)

    targets = _group_targets(
        len(group_names), train_fraction, validation_fraction, test_group_count
    )
    train_groups = targets["train"]
    validation_groups = targets["validation"]
    test_groups = targets["test"]

    assignments = {
        "train": set(group_names[:train_groups]),
        "validation": set(group_names[train_groups:train_groups + validation_groups]),
        "test": set(group_names[-test_groups:]),
    }
    return {
        split: [record for record in records if record.record_group in selected]
        for split, selected in assignments.items()
    }


def stratified_grouped_split(
    records: Sequence[SampleRecord],
    group_scores: dict[str, float],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 20260809,
    test_group_count: int | None = None,
) -> dict[str, list[SampleRecord]]:
    """Group-safe split that also balances a per-group score across splits.

    Same leakage guarantee as ``grouped_split`` (a whole ground-motion pair
    goes to exactly one split) but, instead of a single random shuffle,
    orders groups by ``group_scores`` and hands them out with a deficit
    round-robin so every split gets a proportional spread of low-to-high
    scores. With only a few dozen distinct ground-motion pairs, a plain
    random shuffle can easily give one split (typically validation, since
    it is small) a subset that is systematically more or less severe than
    train purely by chance -- see ``load_group_intensity_scores``.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than one.")

    groups: dict[str, list[SampleRecord]] = {}
    for record in records:
        groups.setdefault(record.record_group, []).append(record)
    group_names = sorted(groups)
    if len(group_names) < 3:
        raise ValueError("At least three distinct ground-motion groups are required.")
    missing = [name for name in group_names if name not in group_scores]
    if missing:
        raise ValueError(
            f"Missing intensity scores for {len(missing)} record group(s); "
            "rebuild the engineered-feature cache before stratifying a split."
        )

    targets = _group_targets(
        len(group_names), train_fraction, validation_fraction, test_group_count
    )

    # Sort low -> high score; a seeded tie-breaker keeps the order
    # deterministic without letting exact ties fall back on group name.
    jitter = random.Random(seed)
    ordered = sorted(group_names, key=lambda name: (group_scores[name], jitter.random()))

    assigned: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    counts = {name: 0 for name in targets}
    for group_name in ordered:
        split_name = max(
            targets,
            key=lambda name: (
                (targets[name] - counts[name]) / targets[name] if targets[name] else -1.0
            ),
        )
        assigned[split_name].append(group_name)
        counts[split_name] += 1

    assignments = {name: set(names) for name, names in assigned.items()}
    return {
        split: [record for record in records if record.record_group in selected]
        for split, selected in assignments.items()
    }


def write_split_manifest(path: str | Path, splits: dict[str, Sequence[SampleRecord]]) -> None:
    payload = {
        "grouping": "ordered record_id_x|record_id_y pair",
        "splits": {
            name: [
                {
                    **asdict(record),
                    "path": str(record.path),
                    "record_group": record.record_group,
                }
                for record in records
            ]
            for name, records in splits.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class _RunningMoments:
    def __init__(self, feature_count: int):
        self.count = 0
        self.total = np.zeros(feature_count, dtype=np.float64)
        self.total_sq = np.zeros(feature_count, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.total.size)
        self.count += values.shape[0]
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)

    def finish(self) -> tuple[list[float], list[float]]:
        if self.count == 0:
            raise ValueError("Cannot calculate normalization from an empty dataset.")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[std < 1.0e-8] = 1.0
        return mean.tolist(), std.tolist()


@dataclass
class NormalizationStats:
    mean: dict[str, list[float]]
    std: dict[str, list[float]]

    @classmethod
    def calculate(
        cls, records: Sequence[SampleRecord], engineered_features=None
    ) -> "NormalizationStats":
        dimensions = {
            "x": 8,
            "edge_attr": 14,
            "global_features": 43,
            "record_features": 12,
            "ground_motion": 2,
            "response": 2,
        }
        if engineered_features is not None:
            dimensions["engineered_features"] = engineered_features.dimension
        running = {key: _RunningMoments(size) for key, size in dimensions.items()}
        for record in records:
            with np.load(record.path, allow_pickle=False) as sample:
                running["x"].update(sample["x"])
                running["edge_attr"].update(sample["edge_attr"])
                running["global_features"].update(sample["global_features"])
                running["record_features"].update(sample["record_features"].reshape(1, -1))
                running["ground_motion"].update(sample["ground_motion"])
                running["response"].update(sample["response_time_history"][:, :2])
            if engineered_features is not None:
                running["engineered_features"].update(
                    engineered_features.vector(record.sample_id)
                )
        mean: dict[str, list[float]] = {}
        std: dict[str, list[float]] = {}
        for key, moments in running.items():
            mean[key], std[key] = moments.finish()
        return cls(mean=mean, std=std)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"mean": self.mean, "std": self.std}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(mean=payload["mean"], std=payload["std"])

    def normalize(self, key: str, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean[key], dtype=np.float32)
        std = np.asarray(self.std[key], dtype=np.float32)
        return (np.asarray(values, dtype=np.float32) - mean) / std

    def target_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.mean["response"], dtype=torch.float32, device=device)
        std = torch.tensor(self.std["response"], dtype=torch.float32, device=device)
        return mean, std


class HybridDataset(Dataset):
    """Lazy NPZ dataset with optional train-only normalization and time stride."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        normalization: NormalizationStats | None = None,
        engineered_features=None,
        sequence_stride: int = 1,
        maximum_steps: int | None = None,
    ):
        if sequence_stride < 1:
            raise ValueError("sequence_stride must be positive.")
        self.records = list(records)
        self.normalization = normalization
        self.engineered_features = engineered_features
        self.sequence_stride = sequence_stride
        self.maximum_steps = maximum_steps

    def __len__(self) -> int:
        return len(self.records)

    def _normal(self, key: str, values: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return np.asarray(values, dtype=np.float32)
        return self.normalization.normalize(key, values)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with np.load(record.path, allow_pickle=False) as sample:
            edge_index = np.asarray(sample["edge_index"], dtype=np.int64)
            x = self._normal("x", sample["x"])
            edge_attr = self._normal("edge_attr", sample["edge_attr"])
            global_features = self._normal("global_features", sample["global_features"])
            record_features = self._normal(
                "record_features", sample["record_features"].reshape(-1)
            )
            stop = None
            if self.maximum_steps is not None:
                stop = self.maximum_steps * self.sequence_stride
            selection = slice(0, stop, self.sequence_stride)
            ground_motion = self._normal("ground_motion", sample["ground_motion"][selection])
            response = self._normal(
                "response", sample["response_time_history"][selection, :2]
            )

        graph = Data(
            x=torch.from_numpy(np.ascontiguousarray(x)),
            edge_index=torch.from_numpy(np.ascontiguousarray(edge_index)),
            edge_attr=torch.from_numpy(np.ascontiguousarray(edge_attr)),
        )
        result = {
            "graph": graph,
            "global_features": torch.from_numpy(np.ascontiguousarray(global_features)),
            "record_features": torch.from_numpy(np.ascontiguousarray(record_features)),
            "ground_motion": torch.from_numpy(np.ascontiguousarray(ground_motion)),
            "target": torch.from_numpy(np.ascontiguousarray(response)),
            "length": int(response.shape[0]),
            "sample_id": record.sample_id,
            "path": str(record.path),
        }
        if self.engineered_features is not None:
            engineered = self._normal(
                "engineered_features",
                self.engineered_features.vector(record.sample_id),
            )
            result["engineered_features"] = torch.from_numpy(
                np.ascontiguousarray(engineered)
            )
        return result


def collate_hybrid(samples: Sequence[dict]) -> dict:
    """Batch variable graphs and pad variable-length ground-motion sequences."""
    lengths = torch.tensor([item["length"] for item in samples], dtype=torch.long)
    ground_motion = pad_sequence(
        [item["ground_motion"] for item in samples], batch_first=True
    )
    target = pad_sequence([item["target"] for item in samples], batch_first=True)
    time_index = torch.arange(ground_motion.shape[1]).unsqueeze(0)
    mask = time_index < lengths.unsqueeze(1)
    result = {
        "graph": Batch.from_data_list([item["graph"] for item in samples]),
        "global_features": torch.stack([item["global_features"] for item in samples]),
        "record_features": torch.stack([item["record_features"] for item in samples]),
        "ground_motion": ground_motion,
        "target": target,
        "lengths": lengths,
        "mask": mask,
        "sample_ids": [item["sample_id"] for item in samples],
        "paths": [item["path"] for item in samples],
    }
    if "engineered_features" in samples[0]:
        result["engineered_features"] = torch.stack(
            [item["engineered_features"] for item in samples]
        )
    return result


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
