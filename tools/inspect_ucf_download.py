import argparse
import os
import zipfile
import tarfile
from pathlib import Path

import numpy as np


def inspect_npy(path):
    try:
        arr = np.load(path, allow_pickle=True)
        print(f"[NPY] {path}")
        print(f"      shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            print("      object array: ilk eleman tipi:", type(arr.flat[0]) if arr.size else None)
    except Exception as e:
        print(f"[NPY ERROR] {path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Downloads içindeki zip/tar/klasör/npy yolu")
    args = parser.parse_args()

    p = Path(args.path).expanduser()

    if not p.exists():
        raise FileNotFoundError(p)

    print(f"İncelenen yol: {p}")
    print(f"Boyut: {p.stat().st_size / (1024 ** 3):.3f} GB" if p.is_file() else "Klasör")

    if p.is_file() and p.suffix.lower() == ".zip":
        print("\nZIP içeriği ilk 80 dosya:")
        with zipfile.ZipFile(p, "r") as z:
            names = z.namelist()
            for name in names[:80]:
                print(" ", name)
            print(f"Toplam dosya: {len(names)}")
        return

    if p.is_file() and (p.suffix.lower() in [".tar", ".gz", ".tgz"] or ".tar" in p.name):
        print("\nTAR içeriği ilk 80 dosya:")
        with tarfile.open(p, "r:*") as t:
            members = t.getmembers()
            for m in members[:80]:
                print(" ", m.name)
            print(f"Toplam dosya: {len(members)}")
        return

    if p.is_file() and p.suffix.lower() == ".npy":
        inspect_npy(p)
        return

    if p.is_dir():
        files = list(p.rglob("*"))
        print(f"\nToplam öğe: {len(files)}")

        npys = [x for x in files if x.suffix.lower() == ".npy"]
        pkls = [x for x in files if x.suffix.lower() in [".pkl", ".pt", ".pth"]]
        mp4s = [x for x in files if x.suffix.lower() in [".mp4", ".avi", ".mkv"]]
        txts = [x for x in files if x.suffix.lower() in [".txt", ".list"]]

        print(f".npy sayısı: {len(npys)}")
        print(f"checkpoint benzeri dosya sayısı: {len(pkls)}")
        print(f"video sayısı: {len(mp4s)}")
        print(f"liste/txt sayısı: {len(txts)}")

        print("\nİlk 20 .npy:")
        for x in npys[:20]:
            print(" ", x)
        print("\nİlk 20 checkpoint benzeri:")
        for x in pkls[:20]:
            print(" ", x)
        print("\nİlk 20 liste/txt:")
        for x in txts[:20]:
            print(" ", x)

        print("\nİlk 5 .npy shape:")
        for x in npys[:5]:
            inspect_npy(x)


if __name__ == "__main__":
    main()
