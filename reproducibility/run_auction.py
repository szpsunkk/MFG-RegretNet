"""Launch the unchanged RQ1--RQ3 drivers using the included frozen model."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from runtime import AUCTION, SNAPSHOT, WORK, activate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("rq1", "rq2_timing", "rq3", "collusion", "adaptive", "accountant"))
    args, extra = parser.parse_known_args()
    activate()
    if args.experiment == "rq2_timing":
        extra += ["--code-dir", str(WORK / "frozen"), "--models-dir", str(WORK / "frozen")]
    subprocess.run([sys.executable, str(AUCTION / (args.experiment + "_p2v2.py")), *extra], check=True)


if __name__ == "__main__":
    main()
