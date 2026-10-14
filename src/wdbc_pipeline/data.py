"""Data acquisition, auditing, cleaning, stratified split and the holdout lock.

The holdout is written to disk with its SHA-256 at split time and can only be
read back through :class:`HoldoutGuard`, which

* requires the frozen pre-holdout configuration (final model + threshold),
* verifies the holdout file has not been modified,
* allows a single opening per run (``force=True`` re-opens it, and every
  opening is appended to ``metadata/holdout_access_log.jsonl``).
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .artifacts import RunContext, sha256_bytes, sha256_file
from .config import DataConfig
from .logging_utils import get_logger

log = get_logger("data")

SOURCE_ID = "_source_row_id"
TARGET_ENC = "_target"


def clean_name(x: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z_]+", "_", str(x).strip()).strip("_")).lower()


@dataclass
class Split:
    X: pd.DataFrame
    y: np.ndarray
    ids: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


# --------------------------------------------------------------------------- acquisition
def fetch_raw_bytes(cfg: DataConfig, project_root: Path | None = None) -> tuple[bytes, str]:
    """Return raw CSV bytes and a human-readable source description."""
    if cfg.source == "url":
        assert cfg.url
        with urllib.request.urlopen(cfg.url, timeout=60) as resp:
            return resp.read(), cfg.url
    if cfg.source == "csv":
        path = Path(cfg.csv_path or "")
        if not path.is_absolute() and project_root is not None:
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"data.csv_path not found: {path}")
        return path.read_bytes(), str(path)
    # sklearn: official UCI WDBC copy bundled with scikit-learn (works offline)
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    # sklearn encodes malignant=0, benign=1 -> map to the UCI labels M/B
    frame.insert(0, cfg.target, np.where(frame.pop("target").to_numpy() == 0, "M", "B"))
    buf = io.StringIO()
    frame.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8"), "sklearn.datasets.load_breast_cancer"


def verify_hash(raw: bytes, expected: str | None) -> str:
    digest = sha256_bytes(raw)
    if expected:
        if digest.lower() != expected.lower():
            raise ValueError(
                f"Dataset SHA-256 mismatch: expected {expected}, got {digest}. "
                "The source changed — refusing to continue."
            )
        log.info("Dataset SHA-256 verified (%s…)", digest[:12])
    else:
        log.warning(
            "data.expected_sha256 is not set; recorded %s. Pin it in the config to make the "
            "source immutable.",
            digest,
        )
    return digest


# --------------------------------------------------------------------------- cleaning
def prepare_frame(raw_df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, dict]:
    """Normalise names, drop ID-like columns and duplicates, encode target."""
    df = raw_df.copy()
    df.columns = [clean_name(c) for c in df.columns]
    # Fully empty trailing columns (e.g. Kaggle "Unnamed: 32") carry no information.
    empty = [c for c in df.columns if df[c].isna().all()]
    target = clean_name(cfg.target)
    if target not in df.columns:
        raise KeyError(f"Missing target column '{target}'; columns={df.columns.tolist()}")
    id_like = {clean_name(c) for c in cfg.id_like_columns}
    drop = sorted({c for c in df.columns if (c in id_like or c in empty) and c != target})
    n_missing_target = int(df[target].isna().sum())
    df = df.dropna(subset=[target]).drop(columns=drop)
    duplicates = int(df.duplicated().sum())
    if cfg.drop_duplicates:
        df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    raw_y = df[target].astype(str).str.strip()
    labels = sorted(raw_y.unique())
    if len(labels) != 2 or cfg.positive_label not in labels:
        raise ValueError(
            f"Expected a binary target containing '{cfg.positive_label}', got {labels}"
        )
    negative = next(v for v in labels if v != cfg.positive_label)
    df[TARGET_ENC] = raw_y.map({negative: 0, cfg.positive_label: 1}).astype(int)
    df = df.drop(columns=[target])
    df.insert(0, SOURCE_ID, np.arange(len(df), dtype=int))
    info = {
        "dropped_columns": drop,
        "rows_missing_target": n_missing_target,
        "duplicates_found": duplicates,
        "duplicates_removed": duplicates if cfg.drop_duplicates else 0,
        "labels": {"negative": negative, "positive": cfg.positive_label},
    }
    return df, info


def data_audit(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing_n": df.isna().sum().to_numpy(),
            "unique_n": df.nunique().to_numpy(),
        }
    )


def split_frame(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TARGET_ENC].to_numpy()
    dev_pos, hold_pos = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=y, random_state=seed
    )
    dev, hold = df.iloc[dev_pos].reset_index(drop=True), df.iloc[hold_pos].reset_index(drop=True)
    assert set(dev[SOURCE_ID]).isdisjoint(hold[SOURCE_ID]), "Split is not disjoint"
    assert len(dev) + len(hold) == len(df), "Split is not complete"
    return dev, hold


def to_split(frame: pd.DataFrame) -> Split:
    return Split(
        X=frame.drop(columns=[SOURCE_ID, TARGET_ENC]).reset_index(drop=True),
        y=frame[TARGET_ENC].to_numpy(dtype=int),
        ids=frame[SOURCE_ID].to_numpy(dtype=int),
    )


# --------------------------------------------------------------------------- stage
def run_data_stage(ctx: RunContext, project_root: Path | None = None) -> None:
    cfg = ctx.cfg
    raw, source = fetch_raw_bytes(cfg.data, project_root)
    digest = verify_hash(raw, cfg.data.expected_sha256)
    snapshot = ctx.data / "dataset_snapshot.csv"
    snapshot.write_bytes(raw)
    ctx.save_json(
        {
            "source": source,
            "sha256": digest,
            "bytes": len(raw),
            "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
        "source_manifest",
    )

    raw_df = pd.read_csv(io.BytesIO(raw))
    df, info = prepare_frame(raw_df, cfg.data)
    ctx.save_table(data_audit(df.drop(columns=[SOURCE_ID])), "data_audit")

    dev, hold = split_frame(df, cfg.data.test_size, cfg.project.seed)
    dev_path, hold_path = ctx.data / "development.csv", ctx.data / "holdout_locked.csv"
    dev.to_csv(dev_path, index=False)
    hold.to_csv(hold_path, index=False)

    membership = pd.DataFrame(
        {
            SOURCE_ID: np.r_[dev[SOURCE_ID], hold[SOURCE_ID]],
            "partition": ["development"] * len(dev) + ["locked_internal_holdout"] * len(hold),
        }
    ).sort_values(SOURCE_ID)
    ctx.save_table(membership, "split_membership")

    def counts(s: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}

    manifest = {
        "original_shape": list(raw_df.shape),
        "n_total": len(df),
        "n_features": df.shape[1] - 2,
        "n_development": len(dev),
        "n_holdout": len(hold),
        "development_counts": counts(dev[TARGET_ENC]),
        "holdout_counts": counts(hold[TARGET_ENC]),
        "seed": cfg.project.seed,
        "development_sha256": sha256_file(dev_path),
        "holdout_sha256": sha256_file(hold_path),
        "status": "HOLDOUT_LOCKED",
        **info,
    }
    ctx.save_json(manifest, "split_manifest")
    log.info(
        "Data: n=%d (dev=%d, holdout=%d, features=%d, duplicates removed=%d) — holdout locked",
        len(df),
        len(dev),
        len(hold),
        manifest["n_features"],
        info["duplicates_removed"],
    )


def load_development(ctx: RunContext) -> Split:
    path = ctx.data / "development.csv"
    if not path.exists():
        raise FileNotFoundError("Development data not found — run the 'data' stage first.")
    manifest = ctx.load_json("split_manifest")
    if sha256_file(path) != manifest["development_sha256"]:
        raise RuntimeError("development.csv was modified after the split.")
    return to_split(pd.read_csv(path))


class HoldoutAccessError(RuntimeError):
    """Raised when the locked holdout is accessed out of protocol."""


class HoldoutGuard:
    LOCK = "holdout_opened.lock.json"

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.lock_path = ctx.metadata / self.LOCK

    @property
    def is_opened(self) -> bool:
        return self.lock_path.exists()

    def open(self, *, force: bool = False, reason: str = "final evaluation") -> Split:
        ctx = self.ctx
        if not ctx.has_json("frozen_configuration_before_holdout"):
            raise HoldoutAccessError(
                "The holdout can only be opened after the final model and threshold are frozen "
                "(run the 'final' stage first)."
            )
        if self.is_opened and not force:
            prev = json.loads(self.lock_path.read_text(encoding="utf-8"))
            raise HoldoutAccessError(
                f"The locked holdout was already opened at {prev['opened_utc']}. Re-opening it "
                "would turn it into a tuning set. Pass --force-holdout to re-open deliberately "
                "(the access is logged)."
            )
        path = ctx.data / "holdout_locked.csv"
        manifest = ctx.load_json("split_manifest")
        if sha256_file(path) != manifest["holdout_sha256"]:
            raise HoldoutAccessError("holdout_locked.csv was modified after the split.")
        frozen = ctx.load_json("frozen_configuration_before_holdout")
        entry = {
            "opened_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "reason": reason,
            "forced_reopen": bool(self.is_opened),
            "frozen_model": frozen.get("final_model"),
            "frozen_model_sha256": frozen.get("model_sha256"),
            "frozen_threshold": frozen.get("secondary_oof_threshold"),
        }
        with (ctx.metadata / "holdout_access_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        if not self.is_opened:
            self.lock_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        if entry["forced_reopen"]:
            log.warning("Holdout RE-OPENED with --force-holdout (logged).")
        else:
            log.info("Holdout opened for the first and only time (logged).")
        return to_split(pd.read_csv(path))

    def read_after_opening(self) -> Split:
        """Read access for post-holdout stages (uncertainty, curves, explain)."""
        if not self.is_opened:
            raise HoldoutAccessError("Run the 'holdout' stage before post-holdout analyses.")
        return to_split(pd.read_csv(self.ctx.data / "holdout_locked.csv"))
