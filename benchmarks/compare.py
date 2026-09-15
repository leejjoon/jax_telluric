"""Check benchmark numerical equivalence and an explicit throughput target."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=1.5)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=2e-12)
    args = parser.parse_args()
    old = json.loads(args.baseline.read_text())
    new = json.loads(args.candidate.read_text())
    assert old["case"] == new["case"], "benchmark workloads differ"
    with np.load(args.baseline.with_suffix(".npz")) as reference, np.load(
        args.candidate.with_suffix(".npz")
    ) as candidate:
        for name in ("flux", "gradient"):
            np.testing.assert_allclose(candidate[name], reference[name], rtol=args.rtol, atol=args.atol)
            print(name, "maximum absolute difference:", float(np.max(np.abs(candidate[name] - reference[name]))))
    for name in ("forward", "value_and_gradient"):
        speedup = old[name]["median_ms"] / new[name]["median_ms"]
        print(name, "speedup:", speedup)
        assert speedup >= args.minimum_speedup, f"{name} missed throughput target"


if __name__ == "__main__":
    main()
