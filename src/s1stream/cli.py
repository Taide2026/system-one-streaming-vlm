"""Command line: s1stream run | report | fetch.

  s1stream run <config.yaml> [--out <dir>] [--limit-windows N]
      default --out: results/<config name>/<YYYYmmdd-HHMMSS>
  s1stream report <run dir>
  s1stream fetch [--check] <asset> [<asset> ...]
      fetch: assets.fetch(lock[name], asset_root() / name); --check: assets.check only.
Exit code 0 on success, 1 on an AssetError, ConfigError or FileExistsError (message on stderr).
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Parse argv with argparse and run the subcommand. Import s1stream.plugins before run."""
    raise NotImplementedError
