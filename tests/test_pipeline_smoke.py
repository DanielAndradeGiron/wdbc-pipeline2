import json

import pytest

from wdbc_pipeline.data import HoldoutAccessError
from wdbc_pipeline.pipeline import run_pipeline


@pytest.mark.slow
def test_end_to_end_tiny(tiny_cfg, tmp_path):
    run_dir = tmp_path / "run"
    ctx = run_pipeline(tiny_cfg, run_dir=run_dir, make_zip=True)
    gates = json.loads((ctx.metadata / "quality_checks.json").read_text())
    assert gates["all_required_gates_passed"] is True
    assert gates["informational"]["holdout_openings"] == 1
    assert (run_dir / "TECHNICAL_RESULTS_REPORT.md").exists()
    assert (run_dir / "PUBLICATION_RESULTS_REPORT.docx").exists()
    assert run_dir.with_suffix(".zip").exists()

    # The holdout cannot be silently re-opened ...
    with pytest.raises(HoldoutAccessError):
        run_pipeline(tiny_cfg, stages=["holdout"], run_dir=run_dir)
    # ... but a deliberate, logged re-opening is possible.
    run_pipeline(tiny_cfg, stages=["holdout"], run_dir=run_dir, force_holdout=True, make_zip=False)
    log = (ctx.metadata / "holdout_access_log.jsonl").read_text().strip().splitlines()
    assert len(log) == 2 and json.loads(log[1])["forced_reopen"] is True


@pytest.mark.slow
def test_benchmark_resumes_from_checkpoints(tiny_cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(tiny_cfg, stages=["data", "benchmark"], run_dir=run_dir, make_zip=False)
    ckpts = sorted((run_dir / "checkpoints" / "benchmark").glob("*.joblib"))
    assert len(ckpts) == tiny_cfg.cv.outer_splits * tiny_cfg.cv.outer_repeats
    before = (run_dir / "tables" / "nested_repeated_cv_predictions.csv").read_text()
    run_pipeline(tiny_cfg, stages=["benchmark"], run_dir=run_dir, make_zip=False)
    assert (run_dir / "tables" / "nested_repeated_cv_predictions.csv").read_text() == before
