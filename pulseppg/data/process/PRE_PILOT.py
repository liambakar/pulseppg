"""
Preprocess unlabeled pre-pilot CSV recordings into PulsePPG SSL folders.

The output follows the same hierarchy as DUMMY.py:

    pre_pilot_ssl/
        train/session_id/recording_0/ts_000000.npy
        val/session_id/recording_0/ts_000000.npy
        test/session_id/recording_0/ts_000000.npy

Each numpy file is a float32 array with shape (time, 1), ready for
SSLDataConfig/Mask_DatasetConfig and run_exp.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import json
import shutil
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.signal import filtfilt, resample_poly
from tqdm import tqdm


DATASET_ROOT = Path(__file__).resolve().parents[1] / "datasets"
DEFAULT_RAW_DIR = DATASET_ROOT / "pre_pilot"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "pre_pilot_ssl"
DEFAULT_EMBEDDING_DIR = DATASET_ROOT / "pre_pilot_outputs"
ADC_MAX = 4_194_303.0


@dataclass
class PrePilotPreprocessConfig:
    raw_dir: str = str(DEFAULT_RAW_DIR)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    target_fs: float = 50.0
    chunk_seconds: float = 30.0
    signal_column: str = "auto"
    time_column: str = "timestamp"
    time_unit: str = "auto"
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 1234
    min_chunks_per_session: int = 2
    filter_low_hz: float = 0.5
    filter_high_hz: float = 12.0
    max_saturation_fraction: float = 0.25
    overwrite: bool = False


def get_pre_pilot_ssl_data_config(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> object:
    _ensure_repo_on_path()
    from pulseppg.data.Base_Dataset import SSLDataConfig

    return SSLDataConfig(
        data_folder=str(output_dir),
        data_normalizer_path=False,
        data_clipping=False,
    )


def get_pre_pilot_mask_data_config(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    mask_extended: int = 300,
) -> object:
    _ensure_repo_on_path()
    from pulseppg.models.MotifDist.MotifDist_Model import Mask_DatasetConfig

    return Mask_DatasetConfig(
        data_folder=str(output_dir),
        data_normalizer_path=False,
        data_clipping=False,
        mask_extended=mask_extended,
    )


def preprocess_pre_pilot(
    config: PrePilotPreprocessConfig | None = None,
    **overrides,
) -> Dict[str, object]:
    config = config or PrePilotPreprocessConfig()
    if overrides:
        config = PrePilotPreprocessConfig(**{**asdict(config), **overrides})

    raw_dir = Path(config.raw_dir)
    output_dir = Path(config.output_dir)
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass overwrite=True or --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_sessions = []
    for csv_path in csv_paths:
        chunks, metadata = preprocess_pre_pilot_csv(csv_path, config)
        if len(chunks) < config.min_chunks_per_session:
            metadata["skipped"] = True
            metadata["skip_reason"] = (
                f"only {len(chunks)} chunks; need {config.min_chunks_per_session}"
            )
            processed_sessions.append((csv_path, [], metadata))
            continue
        metadata["skipped"] = False
        metadata["skip_reason"] = ""
        processed_sessions.append((csv_path, chunks, metadata))

    valid_sessions = [(p, chunks, meta) for p, chunks, meta in processed_sessions if chunks]
    split_by_session = _assign_splits(
        [(path, len(chunks)) for path, chunks, _ in valid_sessions],
        ratios=config.split_ratios,
        seed=config.seed,
    )

    metadata_rows = []
    split_counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    for csv_path, chunks, metadata in processed_sessions:
        session_id = _session_id(csv_path)
        split = split_by_session.get(csv_path, "skipped")
        if split == "skipped":
            split_counts["skipped"] += 1
        else:
            session_dir = output_dir / split / session_id / "recording_0"
            session_dir.mkdir(parents=True, exist_ok=True)
            for chunk_idx, chunk in enumerate(chunks):
                np.save(session_dir / f"ts_{chunk_idx:06d}.npy", chunk)
            split_counts[split] += len(chunks)

        metadata_rows.append(
            {
                "file": csv_path.name,
                "session_id": session_id,
                "split": split,
                "num_chunks": len(chunks),
                **metadata,
            }
        )

    _write_metadata(output_dir, metadata_rows, config, split_counts)

    return {
        "output_dir": str(output_dir),
        "num_csv_files": len(csv_paths),
        "num_sessions": len(valid_sessions),
        "split_counts": split_counts,
        "metadata_csv": str(output_dir / "metadata.csv"),
        "manifest_json": str(output_dir / "manifest.json"),
    }


def preprocess_pre_pilot_csv(
    csv_path: str | Path,
    config: PrePilotPreprocessConfig | None = None,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    config = config or PrePilotPreprocessConfig()
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    source_fs = infer_source_fs(
        df,
        time_column=config.time_column,
        time_unit=config.time_unit,
    )
    signal_column = select_ppg_column(
        df,
        signal_column=config.signal_column,
        max_saturation_fraction=config.max_saturation_fraction,
    )
    raw_signal = clean_ppg_column(
        df[signal_column].to_numpy(dtype=float),
        max_saturation_fraction=config.max_saturation_fraction,
    )

    filter_high_hz = min(config.filter_high_hz, source_fs * 0.45)
    filter_low_hz = min(config.filter_low_hz, filter_high_hz * 0.5)
    if filter_high_hz > filter_low_hz and len(raw_signal) > 30:
        filtered = filter_ppg_signal(
            raw_signal,
            frequency=source_fs,
            low_hz=filter_low_hz,
            high_hz=filter_high_hz,
        )
    else:
        filtered = raw_signal

    resampled = resample_signal(
        filtered,
        fs_original=source_fs,
        fs_target=config.target_fs,
    )
    resampled = _zscore(resampled)
    chunks = chunk_signal(
        resampled,
        chunk_samples=round(config.chunk_seconds * config.target_fs),
    )

    metadata = {
        "source_rows": int(len(df)),
        "source_fs": float(source_fs),
        "target_fs": float(config.target_fs),
        "chunk_seconds": float(config.chunk_seconds),
        "signal_column": signal_column,
        "resampled_samples": int(len(resampled)),
    }
    return chunks, metadata


def export_pre_pilot_embeddings(
    model_config_key: str = "pulseppg",
    checkpoint: str = "best",
    processed_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_EMBEDDING_DIR,
    batch_size: int = 128,
    device: str | None = None,
    preprocess_if_missing: bool = True,
    preprocess_overwrite: bool = False,
    max_chunks: int | None = None,
) -> Dict[str, object]:
    _ensure_repo_on_path()

    import torch

    from pulseppg.experiments.configs.PulsePPG_expconfigs import allpulseppg_expconfigs
    from pulseppg.utils.imports import import_net

    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)

    if preprocess_overwrite or (
        preprocess_if_missing and not any(processed_dir.rglob("*.npy"))
    ):
        preprocess_pre_pilot(
            PrePilotPreprocessConfig(
                output_dir=str(processed_dir),
                overwrite=preprocess_overwrite,
            )
        )

    files = sorted(processed_dir.glob("*/*/*/*.npy"))
    if not files:
        raise FileNotFoundError(f"No processed .npy files found in {processed_dir}")
    if max_chunks is not None:
        files = files[:max_chunks]

    if model_config_key not in allpulseppg_expconfigs:
        raise KeyError(f"Unknown PulsePPG config: {model_config_key}")

    model_config = allpulseppg_expconfigs[model_config_key]
    run_dir = Path("pulseppg/experiments/out") / model_config_key
    checkpoint_path = run_dir / f"checkpoint_{checkpoint}.pkl"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {checkpoint_path}. Download weights or train the base model first."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    net = import_net(model_config.net_config).to(torch_device)
    state_dict = torch.load(checkpoint_path, map_location=torch_device)
    net.load_state_dict(state_dict["net"])
    net.eval()

    embeddings = []
    rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(files), batch_size), desc="Embedding pre-pilot"):
            batch_files = files[start : start + batch_size]
            batch_np = np.stack([np.load(path).astype(np.float32) for path in batch_files])
            batch = torch.from_numpy(batch_np).transpose(1, 2).to(torch_device)
            batch_embeddings = net(batch).cpu().numpy()
            embeddings.append(batch_embeddings)

            for offset, path in enumerate(batch_files):
                relative_parts = path.relative_to(processed_dir).parts
                rows.append(
                    {
                        "embedding_index": start + offset,
                        "split": relative_parts[0],
                        "session_id": relative_parts[1],
                        "recording_id": relative_parts[2],
                        "chunk_file": relative_parts[3],
                        "path": str(path),
                    }
                )

    embeddings = np.concatenate(embeddings, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / f"{model_config_key}_{checkpoint}_embeddings.npz"
    index_path = output_dir / f"{model_config_key}_{checkpoint}_embedding_index.csv"

    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        filepaths=np.array([row["path"] for row in rows]),
    )
    pd.DataFrame(rows).to_csv(index_path, index=False)

    return {
        "checkpoint": str(checkpoint_path),
        "processed_dir": str(processed_dir),
        "embeddings_path": str(embeddings_path),
        "index_path": str(index_path),
        "num_chunks": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "device": str(torch_device),
    }


def cluster_pre_pilot_embeddings(
    embeddings_path: str | Path = DEFAULT_EMBEDDING_DIR / "pulseppg_best_embeddings.npz",
    index_path: str | Path = DEFAULT_EMBEDDING_DIR / "pulseppg_best_embedding_index.csv",
    output_dir: str | Path = DEFAULT_EMBEDDING_DIR / "clusters",
    n_clusters: int = 6,
    chunk_seconds: float = 30.0,
    random_state: int = 1234,
) -> Dict[str, object]:
    from scipy.cluster.vq import kmeans2
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = Path(embeddings_path)
    index_path = Path(index_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing embedding index CSV: {index_path}")

    z = np.load(embeddings_path)
    embeddings = z["embeddings"]
    index = pd.read_csv(index_path)
    if len(index) != len(embeddings):
        raise ValueError(
            f"Index rows ({len(index)}) do not match embeddings ({len(embeddings)})"
        )

    n_clusters = min(n_clusters, len(embeddings))
    scaled = StandardScaler().fit_transform(embeddings)
    _, labels = kmeans2(scaled, n_clusters, minit="++", seed=random_state)

    pca_dims = min(2, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=pca_dims, random_state=random_state)
    pca_values = pca.fit_transform(scaled)
    if pca_dims == 1:
        pca_values = np.column_stack([pca_values[:, 0], np.zeros(len(pca_values))])

    labeled = index.copy()
    labeled["cluster"] = labels
    labeled["chunk_index"] = labeled["chunk_file"].map(_chunk_index_from_name)
    labeled["time_minutes"] = labeled["chunk_index"] * chunk_seconds / 60.0
    labeled["pca_1"] = pca_values[:, 0]
    labeled["pca_2"] = pca_values[:, 1]

    labeled_path = output_dir / "pre_pilot_embedding_clusters.csv"
    pca_path = output_dir / "pre_pilot_cluster_pca.png"
    timeline_dir = output_dir / "timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(labeled_path, index=False)

    _save_cluster_pca_plot(labeled, pca_path, n_clusters)
    timeline_paths = _save_cluster_timeline_plots(
        labeled,
        timeline_dir=timeline_dir,
        n_clusters=n_clusters,
    )

    counts_path = output_dir / "pre_pilot_cluster_counts.csv"
    counts = (
        labeled.groupby(["session_id", "cluster"])
        .size()
        .rename("num_chunks")
        .reset_index()
    )
    counts["minutes"] = counts["num_chunks"] * chunk_seconds / 60.0
    counts.to_csv(counts_path, index=False)

    return {
        "embeddings_path": str(embeddings_path),
        "index_path": str(index_path),
        "labeled_csv": str(labeled_path),
        "cluster_counts_csv": str(counts_path),
        "pca_plot": str(pca_path),
        "timeline_dir": str(timeline_dir),
        "num_timeline_plots": len(timeline_paths),
        "num_chunks": int(len(labeled)),
        "n_clusters": int(n_clusters),
    }


def _chunk_index_from_name(chunk_file: str) -> int:
    stem = Path(chunk_file).stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return 0


def _get_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/matplotlib")))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _save_cluster_pca_plot(
    labeled: pd.DataFrame,
    output_path: Path,
    n_clusters: int,
) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        labeled["pca_1"],
        labeled["pca_2"],
        c=labeled["cluster"],
        cmap="tab10",
        s=12,
        alpha=0.8,
        vmin=0,
        vmax=max(n_clusters - 1, 1),
    )
    ax.set_title("Pre-pilot PulsePPG Embedding Clusters")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_clusters))
    cbar.set_label("Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_cluster_timeline_plots(
    labeled: pd.DataFrame,
    timeline_dir: Path,
    n_clusters: int,
) -> List[str]:
    plt = _get_pyplot()
    paths = []
    for session_id, session_df in labeled.groupby("session_id", sort=True):
        session_df = session_df.sort_values("chunk_index")
        fig_height = max(2.4, min(5.0, 1.3 + 0.08 * len(session_df)))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        scatter = ax.scatter(
            session_df["time_minutes"],
            np.zeros(len(session_df)),
            c=session_df["cluster"],
            cmap="tab10",
            s=36,
            marker="s",
            vmin=0,
            vmax=max(n_clusters - 1, 1),
        )
        ax.set_title(f"Cluster Timeline: {session_id}")
        ax.set_xlabel("Time (minutes)")
        ax.set_yticks([])
        ax.set_ylim(-0.8, 0.8)
        ax.grid(axis="x", alpha=0.25)
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_clusters), pad=0.02)
        cbar.set_label("Cluster")
        fig.tight_layout()
        path = timeline_dir / f"{session_id}_cluster_timeline.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def filter_ppg_signal(
    waveform: np.ndarray,
    frequency: float,
    low_hz: float = 0.5,
    high_hz: float = 12.0,
    order: int = 4,
) -> np.ndarray:
    if low_hz <= 0:
        b, a = scipy_signal.cheby2(order, 20, [high_hz], "low", fs=frequency)
    else:
        b, a = scipy_signal.cheby2(order, 20, [low_hz, high_hz], "bandpass", fs=frequency)

    try:
        filtered = filtfilt(b, a, waveform)
    except ValueError:
        return waveform

    if frequency >= 75:
        win = max(1, round(frequency * 50 / 1000))
        kernel = np.ones(win) / win
        filtered = filtfilt(kernel, 1, filtered)
    return filtered


def resample_signal(
    signal: np.ndarray,
    fs_original: float,
    fs_target: float,
) -> np.ndarray:
    if fs_original == fs_target:
        return np.asarray(signal)

    fs_original_frac = Fraction(float(fs_original)).limit_denominator(1000)
    fs_target_frac = Fraction(float(fs_target)).limit_denominator(1000)
    lcm_denominator = np.lcm(fs_original_frac.denominator, fs_target_frac.denominator)
    fs_original_scaled = fs_original_frac * lcm_denominator
    fs_target_scaled = fs_target_frac * lcm_denominator
    gcd_value = gcd(fs_original_scaled.numerator, fs_target_scaled.numerator)
    up = fs_target_scaled.numerator // gcd_value
    down = fs_original_scaled.numerator // gcd_value
    return resample_poly(signal, up, down, axis=0)


def infer_source_fs(
    df: pd.DataFrame,
    time_column: str = "timestamp",
    time_unit: str = "auto",
) -> float:
    if time_column not in df.columns:
        raise ValueError(f"Missing time column {time_column!r}")

    timestamps = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        raise ValueError(f"Could not infer sampling rate from {time_column!r}")

    median_diff = float(np.median(diffs))
    if time_unit == "auto":
        if median_diff > 10_000:
            seconds = median_diff / 1_000_000.0
        elif median_diff > 1:
            seconds = median_diff / 1_000.0
        else:
            seconds = median_diff
    elif time_unit == "us":
        seconds = median_diff / 1_000_000.0
    elif time_unit == "ms":
        seconds = median_diff / 1_000.0
    elif time_unit == "s":
        seconds = median_diff
    else:
        raise ValueError("time_unit must be one of: auto, us, ms, s")

    if seconds <= 0:
        raise ValueError(f"Invalid median timestamp interval: {median_diff}")
    return 1.0 / seconds


def select_ppg_column(
    df: pd.DataFrame,
    signal_column: str = "auto",
    candidates: Sequence[str] = ("green", "ir", "red"),
    max_saturation_fraction: float = 0.25,
) -> str:
    if signal_column != "auto":
        if signal_column not in df.columns:
            raise ValueError(f"Missing requested signal column {signal_column!r}")
        return signal_column

    scores = []
    for column in candidates:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        saturation_fraction = np.mean((finite <= 0) | (finite >= ADC_MAX))
        cleaned = finite[(finite > 0) & (finite < ADC_MAX)]
        std = float(np.std(cleaned)) if len(cleaned) else 0.0
        usable = saturation_fraction <= max_saturation_fraction and std > 0
        scores.append((not usable, saturation_fraction, -std, column))

    if not scores:
        raise ValueError("No usable PPG optical column found")

    scores.sort()
    return scores[0][-1]


def clean_ppg_column(
    values: np.ndarray,
    max_saturation_fraction: float = 0.25,
) -> np.ndarray:
    values = values.astype(float)
    saturated = (values <= 0) | (values >= ADC_MAX)
    if float(np.mean(saturated)) <= max_saturation_fraction:
        values[saturated] = np.nan
    values[~np.isfinite(values)] = np.nan

    if np.all(np.isnan(values)):
        raise ValueError("Selected PPG column has no finite unsaturated samples")

    series = pd.Series(values).interpolate(limit_direction="both")
    return series.to_numpy(dtype=float)


def chunk_signal(signal: np.ndarray, chunk_samples: int) -> List[np.ndarray]:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    usable_samples = (len(signal) // chunk_samples) * chunk_samples
    if usable_samples == 0:
        return []
    signal = signal[:usable_samples].reshape(-1, chunk_samples)
    return [chunk.astype(np.float32)[:, None] for chunk in signal]


def _zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    mean = float(np.nanmean(signal))
    std = float(np.nanstd(signal))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (signal - mean) / std


def _assign_splits(
    session_info: Sequence[Tuple[Path, int]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[Path, str]:
    if not session_info:
        return {}

    rng = np.random.default_rng(seed)
    entries = list(session_info)
    rng.shuffle(entries)

    n_sessions = len(entries)
    if n_sessions < 3:
        return {path: "train" for path, _ in entries}

    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    total_chunks = sum(count for _, count in entries)
    val_target = total_chunks * val_ratio / total
    test_target = total_chunks * test_ratio / total

    remaining = entries.copy()
    test_path = _pop_closest_chunk_count(remaining, test_target)
    val_path = _pop_closest_chunk_count(remaining, val_target)

    split_by_session = {path: "train" for path, _ in entries}
    split_by_session[test_path] = "test"
    split_by_session[val_path] = "val"
    return split_by_session


def _pop_closest_chunk_count(entries: List[Tuple[Path, int]], target: float) -> Path:
    best_idx = min(
        range(len(entries)),
        key=lambda idx: (abs(entries[idx][1] - target), entries[idx][1]),
    )
    path, _ = entries.pop(best_idx)
    return path


def _ensure_repo_on_path() -> None:
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.append(repo_root)


def _session_id(csv_path: Path) -> str:
    return csv_path.stem.replace(" ", "_")


def _write_metadata(
    output_dir: Path,
    metadata_rows: Iterable[Dict[str, object]],
    config: PrePilotPreprocessConfig,
    split_counts: Dict[str, int],
) -> None:
    rows = list(metadata_rows)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "config": asdict(config),
        "split_counts": split_counts,
        "metadata": rows,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target_fs", type=float, default=50.0)
    parser.add_argument("--chunk_seconds", type=float, default=30.0)
    parser.add_argument("--signal_column", default="auto")
    parser.add_argument("--time_column", default="timestamp")
    parser.add_argument("--time_unit", choices=["auto", "us", "ms", "s"], default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min_chunks_per_session", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--model_config", default="pulseppg")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--embedding_output_dir", default=str(DEFAULT_EMBEDDING_DIR))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--n_clusters", type=int, default=6)
    parser.add_argument("--cluster_output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preprocess_keys = PrePilotPreprocessConfig.__dataclass_fields__.keys()
    preprocess_kwargs = {key: getattr(args, key) for key in preprocess_keys}
    summary = preprocess_pre_pilot(PrePilotPreprocessConfig(**preprocess_kwargs))
    if args.embed:
        summary["embedding_export"] = export_pre_pilot_embeddings(
            model_config_key=args.model_config,
            checkpoint=args.checkpoint,
            processed_dir=args.output_dir,
            output_dir=args.embedding_output_dir,
            batch_size=args.batch_size,
            device=args.device,
            preprocess_if_missing=False,
            max_chunks=args.max_chunks,
        )
    if args.cluster:
        cluster_output_dir = args.cluster_output_dir
        if cluster_output_dir is None:
            cluster_output_dir = str(Path(args.embedding_output_dir) / "clusters")
        summary["clustering"] = cluster_pre_pilot_embeddings(
            embeddings_path=Path(args.embedding_output_dir)
            / f"{args.model_config}_{args.checkpoint}_embeddings.npz",
            index_path=Path(args.embedding_output_dir)
            / f"{args.model_config}_{args.checkpoint}_embedding_index.csv",
            output_dir=cluster_output_dir,
            n_clusters=args.n_clusters,
            chunk_seconds=args.chunk_seconds,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
