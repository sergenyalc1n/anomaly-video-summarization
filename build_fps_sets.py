"""
fps ablation icin feature setlerini STANDART UCF-Crime bolunmesine (1610 train /
290 annotated test) hazirlar ve BOZUK/YARIM kopyalari onarir.

  1) features_fps8-*  (4 parca) -> data/ucf/features_fps8/  birlestirir.
  2) Mevcut fps2 seti (data/ucf/features) -> fps1'in standart bolunmesini
     taklit ederek data/ucf/features_fps2_std/  olusturur.

ONEMLI: Bu surum "dosya varsa atla" YERINE, hedef dosyanin boyutu kaynakla
ayni mi ve numpy ile TAM okunabiliyor mu diye bakar; degilse yeniden kopyalar.
Boylece daha onceki yarim/bozuk kopyalar onarilir. Idempotent ve tekrar
calistirilabilir.

Kullanim:
    python build_fps_sets.py
"""
import glob
import os
import shutil

import numpy as np

ROOT = "data/ucf"
FPS1 = f"{ROOT}/features_fps1"
FPS2_SRC = f"{ROOT}/features"
FPS2_DST = f"{ROOT}/features_fps2_std"
FPS8_DST = f"{ROOT}/features_fps8"


def is_valid_copy(src, dst):
    """dst, src'nin tam ve saglam bir kopyasi mi?"""
    if not os.path.exists(dst):
        return False
    try:
        if os.path.getsize(dst) != os.path.getsize(src):
            return False
        np.load(dst, allow_pickle=True)  # tam okuma -> truncation yakalar
        return True
    except Exception:
        return False


def ensure_copy(src, dst):
    """Gerekliyse (eksik/boyut farkli/bozuk) src'yi dst'ye kopyalar. True=(yeniden)kopyalandi."""
    if is_valid_copy(src, dst):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def stems_under(path):
    return set(os.path.basename(p)[:-4] for p in glob.glob(f"{path}/**/*.npy", recursive=True))


def merge_fps8():
    parts = sorted(glob.glob(f"{ROOT}/features_fps8-*/features_fps8"))
    if not parts:
        print("[fps8] parca klasoru bulunamadi, atlaniyor.")
        return
    print(f"[fps8] {len(parts)} parca -> {FPS8_DST} (bozuklari onararak)")
    copied = ok = 0
    for part in parts:
        for src in glob.glob(f"{part}/**/*.npy", recursive=True):
            rel = os.path.relpath(src, part)
            dst = os.path.join(FPS8_DST, rel)
            if ensure_copy(src, dst):
                copied += 1
            else:
                ok += 1
            if (copied + ok) % 400 == 0:
                print(f"  ... {copied} (yeniden)kopyalandi, {ok} zaten saglam")
    tr = len(glob.glob(f"{FPS8_DST}/train/**/*.npy", recursive=True))
    te = len(glob.glob(f"{FPS8_DST}/test/**/*.npy", recursive=True))
    print(f"[fps8] bitti: (yeniden)kopyalanan={copied} saglam={ok} | train={tr} test={te}")


def build_fps2_std():
    if not os.path.isdir(FPS1):
        print(f"[fps2_std] {FPS1} yok, atlaniyor.")
        return
    fps2_map = {os.path.basename(p)[:-4]: p
                for p in glob.glob(f"{FPS2_SRC}/**/*.npy", recursive=True)}
    print(f"[fps2_std] fps2 kaynak={len(fps2_map)} | fps1 sablonu (bozuklari onararak)")
    copied = ok = missing = 0
    for p in glob.glob(f"{FPS1}/**/*.npy", recursive=True):
        rel = os.path.relpath(p, FPS1)
        stem = os.path.basename(p)[:-4]
        src = fps2_map.get(stem)
        if src is None:
            missing += 1
            continue
        dst = os.path.join(FPS2_DST, rel)
        if ensure_copy(src, dst):
            copied += 1
        else:
            ok += 1
        if (copied + ok) % 400 == 0:
            print(f"  ... {copied} (yeniden)kopyalandi, {ok} zaten saglam")
    tr = len(glob.glob(f"{FPS2_DST}/train/**/*.npy", recursive=True))
    te = len(glob.glob(f"{FPS2_DST}/test/**/*.npy", recursive=True))
    print(f"[fps2_std] bitti: (yeniden)kopyalanan={copied} saglam={ok} eksik={missing} | train={tr} test={te}")


def final_integrity_scan():
    """Iki uretilen sette TUM dosyalari np.load ile dene; bozuk kalan var mi raporla."""
    print("\n=== BUTUNLUK TARAMASI (tum dosyalar np.load) ===")
    bad_total = 0
    for name, root in [("fps2_std", FPS2_DST), ("fps8", FPS8_DST)]:
        files = glob.glob(f"{root}/**/*.npy", recursive=True)
        bad = []
        for f in files:
            try:
                np.load(f, allow_pickle=True)
            except Exception:
                bad.append(f)
        bad_total += len(bad)
        print(f"  {name:9s}: {len(files)} dosya, BOZUK={len(bad)}")
        for b in bad[:5]:
            print(f"      ! {b}")
    if bad_total == 0:
        print("  -> Hepsi saglam. Egitime hazir.")
    else:
        print(f"  -> {bad_total} bozuk dosya KALDI. build_fps_sets.py'yi tekrar calistir.")


def verify_splits():
    print("\n=== DOGRULAMA: train setleri ayni mi? ===")
    sets = {
        "fps1": stems_under(f"{FPS1}/train"),
        "fps2_std": stems_under(f"{FPS2_DST}/train"),
        "fps4": stems_under(f"{ROOT}/features_fps4/train"),
        "fps8": stems_under(f"{FPS8_DST}/train"),
    }
    base = sets["fps1"]
    for name, s in sets.items():
        flag = "OK" if s == base else f"FARKLI (kesisim {len(s & base)}, bu sette {len(s)})"
        print(f"  {name:9s} train={len(s):5d}  {flag}")


if __name__ == "__main__":
    merge_fps8()
    build_fps2_std()
    verify_splits()
    final_integrity_scan()
    print("\nHazir.")
