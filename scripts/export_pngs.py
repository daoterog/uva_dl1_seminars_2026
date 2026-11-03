"""Write the CIFAR-10 training set as 50,000 PNG files, for the small-files case (card 10).

    python scripts/export_pngs.py --out data/cifar10_png             # local disk
    python scripts/export_pngs.py --out /scratch-shared/$USER/cifar10_png

Then train or benchmark with `--overlay configs/failures/10_png_files.yaml
data.png_root=<out>`. Write them to the filesystem you want to measure: the point is
how that filesystem copes with many small files, not PNG decoding as such.
"""

import argparse
from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR10


def main() -> None:
    """Download CIFAR-10 if needed, then write `<out>/train/<index>_<label>.png`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", default="data/", help="where CIFAR-10 is (or goes)")
    parser.add_argument("--out", default="data/cifar10_png", help="output directory")
    args = parser.parse_args()

    train = CIFAR10(args.root, train=True, download=True)
    out = Path(args.out) / "train"
    out.mkdir(parents=True, exist_ok=True)
    for i, (image, label) in enumerate(zip(train.data, train.targets, strict=True)):
        Image.fromarray(image).save(out / f"{i:05d}_{label}.png")
    print(f"wrote {len(train.data)} PNGs to {out}")


if __name__ == "__main__":
    main()
