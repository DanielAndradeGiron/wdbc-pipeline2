"""Command-line interface.

Examples::

    python -m wdbc_pipeline run --config configs/quick.yaml
    python -m wdbc_pipeline run --config configs/default.yaml --stages data benchmark
    python -m wdbc_pipeline run --config configs/default.yaml --run-dir outputs/runs/full_ab12cd34ef56 \
        --stages threshold final holdout uncertainty curves explain report
    python -m wdbc_pipeline run --config configs/default.yaml --set runtime.n_jobs=4 --set cv.outer_repeats=2
    python -m wdbc_pipeline show-config --config configs/quick.yaml
    python -m wdbc_pipeline list-models --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import __version__
from .config import ConfigError, config_hash, load_config
from .pipeline import STAGES


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wdbc-pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--config", "-c", default="configs/default.yaml", help="YAML configuration file"
        )
        sp.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config value, e.g. --set runtime.n_jobs=4 (repeatable)",
        )

    run = sub.add_parser("run", help="run the pipeline (all stages or a subset)")
    common(run)
    run.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        metavar="STAGE",
        help=f"stages to run, in canonical order: {' '.join(STAGES)} (default: all)",
    )
    run.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="existing/explicit run directory (default: <output_dir>/<run_name>_<hash>)",
    )
    run.add_argument(
        "--force-holdout",
        action="store_true",
        help="deliberately RE-OPEN the locked holdout (logged in holdout_access_log.jsonl)",
    )
    run.add_argument(
        "--no-zip", action="store_true", help="do not zip the run directory at the end"
    )
    run.add_argument("-v", "--verbose", action="store_true", help="debug-level console logging")

    show = sub.add_parser("show-config", help="print the resolved configuration and its hash")
    common(show)

    lm = sub.add_parser("list-models", help="list candidate models and dependency availability")
    common(lm)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "show-config":
        print(
            yaml.safe_dump(
                json.loads(json.dumps(cfg.to_dict(), default=list)),
                sort_keys=False,
                allow_unicode=True,
            )
        )
        print(f"# config_hash: {config_hash(cfg)}")
        return 0

    if args.command == "list-models":
        from .models import REGISTRY, missing_dependencies

        for name in REGISTRY:
            miss = missing_dependencies(name)
            flag = "included" if name in cfg.models.include else "excluded"
            state = "OK" if not miss else f"missing: {', '.join(miss)}"
            print(f"{name:<18} {flag:<9} {state}")
        return 0

    from .pipeline import run_pipeline

    project_root = Path(args.config).resolve().parent.parent
    from .data import HoldoutAccessError
    from .report import QualityGateError

    try:
        run_pipeline(
            cfg,
            stages=args.stages,
            run_dir=args.run_dir,
            force_holdout=args.force_holdout,
            project_root=project_root,
            verbose=args.verbose,
            make_zip=not args.no_zip,
        )
    except (
        HoldoutAccessError,
        QualityGateError,
        FileNotFoundError,
        ImportError,
        RuntimeError,
    ) as exc:
        if args.verbose:
            raise
        print(f"\nERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
