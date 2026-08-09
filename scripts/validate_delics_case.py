"""Validate an extracted DeliCS raw-MRF case without loading it into RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qMR_Robust.data.external_mrf import validate_delics_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-coils", type=int, default=48)
    parser.add_argument("--n-tr", type=int, default=500)
    parser.add_argument("--n-repeats", type=int, default=48)
    args = parser.parse_args()

    report = validate_delics_case(
        args.case_dir,
        expected_coils=args.expected_coils,
        n_tr=args.n_tr,
        n_repeats=args.n_repeats,
    )
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
