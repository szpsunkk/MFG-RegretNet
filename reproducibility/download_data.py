"""Download only the public image datasets required by the frozen RQ4 protocol."""
import argparse
from pathlib import Path
from torchvision.datasets import MNIST, CIFAR10

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("data"))
parser.add_argument("--dataset", choices=("MNIST", "CIFAR10", "both"), default="both")
args = parser.parse_args()
for name, cls, subdir in (("MNIST", MNIST, "mnist"), ("CIFAR10", CIFAR10, "cifar10")):
    if args.dataset in (name, "both"):
        for train in (True, False):
            cls(root=str(args.out / subdir), train=train, download=True)
        print(name, "ready:", (args.out / subdir).resolve())
