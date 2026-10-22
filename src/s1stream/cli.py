"""Command line: s1stream run | report | fetch.

  s1stream run <config.yaml> [--out <dir>] [--limit-windows N]
      default --out: results/<config name>/<YYYYmmdd-HHMMSS>
  s1stream report <run dir>
  s1stream fetch [--check] <asset> [<asset> ...]
  s1stream probe-extract <probe config.yaml> --condition <name> --run <name> [--limit-windows N]
      s1stream.probe.extract.run_extract; output $S1_FEATURES/<config name>/<condition>/<run name>
      fetch: assets.fetch(lock[name], asset_root() / name); --check: assets.check only.
Exit code 0 on success, 1 on an AssetError, ConfigError, DatasetError or FileExistsError (message on stderr).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from s1stream import assets
from s1stream.config import ConfigError, load_config
from s1stream.datasets.ucf_crime import DatasetError
from s1stream.report import report
from s1stream.runner import run


def main(argv: list[str] | None = None) -> int:
    """Parse argv with argparse and run the subcommand. Import s1stream.plugins before run."""
    parser = argparse.ArgumentParser(prog="s1stream")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--out", type=Path)
    run_parser.add_argument("--limit-windows", type=int)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("run_dir", type=Path)
    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument("--check", action="store_true")
    fetch_parser.add_argument("assets", nargs="+")
    extract_parser = commands.add_parser("probe-extract")
    extract_parser.add_argument("config", type=Path)
    extract_parser.add_argument("--condition", required=True)
    extract_parser.add_argument("--run", required=True)
    extract_parser.add_argument("--limit-windows", type=int)
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            import s1stream.plugins  # noqa: F401  Register all run plugins.

            config = load_config(args.config)
            out_dir = args.out
            if out_dir is None:
                out_dir = Path("results") / config.name / datetime.now().strftime("%Y%m%d-%H%M%S")
            print(run(config, out_dir, limit_windows=args.limit_windows))
        elif args.command == "probe-extract":
            import s1stream.plugins  # noqa: F401  Register the dataset and input-form plugins.
            from s1stream.probe.extract import run_extract

            print(run_extract(args.config, args.condition, args.run, limit_windows=args.limit_windows))
        elif args.command == "report":
            report(args.run_dir)
        else:
            lock = assets.load_lock(assets.LOCK_FILE)
            for name in args.assets:
                if name not in lock:
                    raise assets.AssetError(f"Asset {name!r} is not in the lock file")
                folder = assets.asset_root() / name
                if args.check:
                    assets.check(lock[name], folder)
                else:
                    assets.fetch(lock[name], folder)
    except (assets.AssetError, ConfigError, DatasetError, FileExistsError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0
