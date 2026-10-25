"""Recompute the 32-run aggregation control from original confusion matrices."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

from runtime import HERE, SNAPSHOT, sha, verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "outputs/aggregation-control")
    args = parser.parse_args()
    verify_release()
    sys.path.insert(0, str(SNAPSHOT / "main_final2/revisions/agg2x2_design_20261023/exp"))
    import agg2x2_analysis as A
    jobs = json.loads((HERE / "evidence/agg2x2_index.json").read_text())
    with zipfile.ZipFile(HERE / "evidence/agg2x2_records.zip") as archive:
        if set(archive.namelist()) != {j["done"] for j in jobs}:
            raise ValueError("unexpected control archive members")
        for j in jobs:
            path = args.out / j["done"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(archive.read(j["done"]))
            if sha(path) != j["result_sha256"]:
                raise ValueError("control result hash mismatch")
            j["done"] = str(path)
    registration = json.loads((HERE / "evidence/agg2x2_registration.json").read_text())
    actual = A.analyse(jobs, registration)
    expected = json.loads((HERE / "evidence/aggregation_2x2.json").read_text())
    expected.pop("inputs_sha256")
    if actual != expected:
        raise AssertionError("control recomputation differs from the paper")
    (args.out / "agg2x2_result.json").write_text(json.dumps(actual, indent=2) + "\n")
    A.write_md(args.out / "agg2x2_result.md", actual)
    print("Verified all 32 control runs; reproduced all three analysis blocks.")


if __name__ == "__main__":
    main()
