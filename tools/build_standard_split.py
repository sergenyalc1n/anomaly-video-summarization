"""
UCF-Crime'in RESMI train/test split'ini kurar.

Mevcut (bozuk) split'teki .npy feature'lari yeniden duzenler:
  - Test  = Anomaly_Test.txt'teki 290 video (140 anomali + 150 normal)
  - Train = geri kalan tum videolar

Orijinal feature'lara DOKUNMAZ; yeni yapiyi data/ucf/features_std/ altinda
hardlink ile olusturur (ekstra disk yer kaplamaz). Hardlink mumkun degilse kopyalar.

Kullanim:
  python tools/build_standard_split.py
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "ucf" / "features"          # mevcut feature'lar (train+test karisik)
DST = ROOT / "data" / "ucf" / "features_std"      # yeni standart split
TEST_LIST = ROOT / "UCF_Crime" / "meta" / "Anomaly_Test.txt"

NORMAL_TRAIN_DIR = "Training-Normal-Videos"
NORMAL_TEST_DIR = "Testing_Normal_Videos_Anomaly"


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)          # hardlink: ekstra yer yok
    except OSError:
        shutil.copy2(src, dst)     # farkli disk vs -> kopya


def main():
    assert SRC.exists(), f"Kaynak feature klasoru yok: {SRC}"
    assert TEST_LIST.exists(), f"Test listesi yok: {TEST_LIST}"

    # 1) Tum mevcut .npy'leri stem -> path olarak topla
    all_npy = {}
    for p in SRC.rglob("*.npy"):
        all_npy[p.stem] = p
    print(f"Toplam feature dosyasi: {len(all_npy)}")

    # 2) Resmi test listesini oku:  "Class/Stem_x264.mp4"
    test_entries = {}   # stem -> class_folder
    for line in TEST_LIST.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls, fname = line.split("/", 1)
        stem = Path(fname).stem
        # Normal test videolarini tek bir standart klasore koy
        if "normal" in cls.lower():
            cls = NORMAL_TEST_DIR
        test_entries[stem] = cls
    print(f"Resmi test videosu: {len(test_entries)}")

    # 3) Yerlestir
    placed_test, placed_train, missing = 0, 0, []

    # 3a) Test
    for stem, cls in test_entries.items():
        src = all_npy.get(stem)
        if src is None:
            missing.append(stem)
            continue
        link_or_copy(src, DST / "test" / cls / f"{stem}.npy")
        placed_test += 1

    # 3b) Train = test'te olmayan her sey
    test_stems = set(test_entries.keys())
    for stem, src in all_npy.items():
        if stem in test_stems:
            continue
        # Sinif klasoru: normalse standart train-normal klasoru, degilse mevcut sinif
        parent = src.parent.name
        cls = NORMAL_TRAIN_DIR if "normal" in parent.lower() else parent
        link_or_copy(src, DST / "train" / cls / f"{stem}.npy")
        placed_train += 1

    # 4) frame_counts.json'u da kopyala (frame-level eval icin lazim)
    fc = SRC / "frame_counts.json"
    if fc.exists():
        link_or_copy(fc, DST / "frame_counts.json")

    # 5) Rapor
    print("\n--- SONUC ---")
    print(f"Train'e yerlesen : {placed_train}")
    print(f"Test'e yerlesen  : {placed_test}")
    if missing:
        print(f"UYARI: test listesinde olup feature'i bulunamayan {len(missing)} video:")
        for m in missing[:20]:
            print("   ", m)

    def count(split):
        base = DST / split
        files = list(base.rglob("*.npy"))
        norm = [f for f in files if "normal" in f.parent.name.lower()]
        return len(files), len(norm), len(files) - len(norm)

    for split in ["train", "test"]:
        t, n, a = count(split)
        print(f"{split}: toplam={t}  normal={n}  anormal={a}")

    print(f"\nYeni split hazir: {DST}")


if __name__ == "__main__":
    main()
