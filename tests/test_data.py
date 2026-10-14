import pandas as pd
import pytest

from wdbc_pipeline.data import (
    HoldoutAccessError,
    HoldoutGuard,
    clean_name,
    load_development,
    prepare_frame,
    run_data_stage,
)


def test_clean_name():
    assert clean_name("Unnamed: 32") == "unnamed_32"
    assert clean_name("mean radius") == "mean_radius"


def test_prepare_frame_drops_ids_duplicates_and_encodes(tiny_cfg):
    raw = pd.DataFrame(
        {
            "ID": [1, 2, 3, 3],
            "diagnosis": ["M", "B", "B", "B"],
            "x": [1.0, 2.0, 3.0, 3.0],
            "Unnamed: 32": [None] * 4,
        }
    )
    raw.loc[3, "ID"] = 3  # same id + same values -> duplicate after dropping ID
    df, info = prepare_frame(raw, tiny_cfg.data)
    assert "id" not in df.columns and "unnamed_32" not in df.columns
    assert info["duplicates_removed"] == 1 and len(df) == 3
    assert df["_target"].tolist() == [1, 0, 0]


def test_prepare_frame_rejects_wrong_labels(tiny_cfg):
    raw = pd.DataFrame({"diagnosis": ["X", "Y"], "x": [1, 2]})
    with pytest.raises(ValueError):
        prepare_frame(raw, tiny_cfg.data)


def test_data_stage_split_is_stratified_and_locked(ctx):
    run_data_stage(ctx)
    m = ctx.load_json("split_manifest")
    assert (m["n_total"], m["n_development"], m["n_holdout"]) == (569, 455, 114)
    assert m["development_counts"] == {"0": 285, "1": 170}
    assert m["holdout_counts"] == {"0": 72, "1": 42}
    assert len(load_development(ctx)) == 455
    with pytest.raises(HoldoutAccessError):  # not frozen yet
        HoldoutGuard(ctx).open()
    with pytest.raises(HoldoutAccessError):
        HoldoutGuard(ctx).read_after_opening()


def test_tampered_development_file_is_detected(ctx):
    run_data_stage(ctx)
    path = ctx.data / "development.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError):
        load_development(ctx)


def test_expected_hash_mismatch_stops(tiny_cfg, tmp_path):
    from wdbc_pipeline.artifacts import RunContext
    from wdbc_pipeline.config import load_config

    from .conftest import ROOT

    cfg = load_config(ROOT / "configs" / "quick.yaml", ["data.expected_sha256=deadbeef"])
    with pytest.raises(ValueError, match="mismatch"):
        run_data_stage(RunContext.create(cfg, tmp_path / "r"))
