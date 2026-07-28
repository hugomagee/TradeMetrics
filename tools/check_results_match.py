"""Compare two engine results JSON files, allowing floating-point tolerance.

    python tools/check_results_match.py outputs/results.json /tmp/fresh.json

Exits non-zero with a description of the first mismatch.

Why a tolerance rather than a byte comparison: the engine is deterministic on a
given machine, but the last bit or two of a float can differ between CPU
architectures and BLAS builds (an x86 CI runner vs an ARM laptop). Demanding
byte-identical output across architectures would fail for reasons that have
nothing to do with correctness. CI therefore checks two things separately:

1. run the demo twice on the same machine and require byte-identical output —
   that is the determinism property worth guarding;
2. compare the committed artifact against a fresh run with this script — that
   is the "committed numbers still match the code" property.
"""

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


def compare(a, b, path: str, rtol: float, atol: float, problems: list) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                problems.append(f"{path}.{key}: missing from first file")
            elif key not in b:
                problems.append(f"{path}.{key}: missing from second file")
            else:
                compare(a[key], b[key], f"{path}.{key}", rtol, atol, problems)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, f"{path}[{i}]", rtol, atol, problems)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            problems.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            problems.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        problems.append(f"{path}: {a!r} vs {b!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    args = parser.parse_args()

    a = json.loads(args.first.read_text())
    b = json.loads(args.second.read_text())

    problems: list[str] = []
    compare(a, b, "results", args.rtol, args.atol, problems)

    if problems:
        print(
            f"{len(problems)} mismatch(es) between {args.first} and {args.second} "
            f"(rtol={args.rtol}, atol={args.atol}):"
        )
        for line in problems[:25]:
            print(f"  {line}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        return 1

    print(f"{args.first} matches {args.second} within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
