#!/usr/bin/env python3
"""Compare the scores of two runs window by window: scripts/compare-runs.py <run A> <run B> [--unevicted N]

The pilot's correctness check: S2 (kv, past_s = L) must give S1's (recompute, past_s = L) probabilities while nothing
has been evicted yet, i.e. for the windows 0 .. N of every video (N = L / stride). Prints the max |p_A - p_B| over
those windows and over all windows, the mean, and the worst video. Clef-Flash runs in bf16: differences around 1e-2
are numerics (a chunked prefill with an initial state vs. one pass); larger ones are a bug.
"""
import argparse
import json
from pathlib import Path


def scores(run: Path) -> dict[tuple[str, int], float]:
    return {(row["video"], row["window"]): row["score"]
            for row in (json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--unevicted", type=int, default=1, help="windows 0..N of a video hold nothing evicted yet")
    args = parser.parse_args()
    a, b = scores(args.a), scores(args.b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no common windows")
    diffs = {key: abs(a[key] - b[key]) for key in common}
    early = {key: d for key, d in diffs.items() if key[1] <= args.unevicted}
    worst = max(diffs, key=diffs.get)
    print(f"{len(common)} common windows ({len(set(a) ^ set(b))} only in one run)")
    print(f"windows 0..{args.unevicted} (nothing evicted): max |dp| = {max(early.values()):.4f} over {len(early)} windows")
    print(f"all windows: max |dp| = {diffs[worst]:.4f} at {worst[0]} window {worst[1]} "
          f"(p {a[worst]:.3f} vs {b[worst]:.3f}), mean |dp| = {sum(diffs.values()) / len(diffs):.4f}")


if __name__ == "__main__":
    main()
