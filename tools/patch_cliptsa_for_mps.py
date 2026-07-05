from pathlib import Path

repo = Path("third_party/CLIP-TSA")

replacements = {
    "dataset.py": {
        "torch.set_default_tensor_type('torch.cuda.FloatTensor')": "torch.set_default_dtype(torch.float32)",
        "np.int": "int",
    },
    "model.py": {
        "torch.set_default_tensor_type('torch.cuda.FloatTensor')": "torch.set_default_dtype(torch.float32)",
        "select_idx = torch.ones_like(nfea_magnitudes).cuda()": "select_idx = torch.ones_like(nfea_magnitudes, device=nfea_magnitudes.device)",
        "select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()": "select_idx_normal = torch.ones_like(nfea_magnitudes, device=nfea_magnitudes.device)",
        "total_select_abn_feature = torch.zeros(0)": "total_select_abn_feature = torch.zeros(0, device=features.device)",
        "total_select_nor_feature = torch.zeros(0)": "total_select_nor_feature = torch.zeros(0, device=features.device)",
    },
    "train.py": {
        "torch.set_default_tensor_type('torch.cuda.FloatTensor')": "torch.set_default_dtype(torch.float32)",
        "label = label.cuda()": "label = label.to(score.device)",
    },
    "utils/utils.py": {
        "dtype=np.int": "dtype=int",
    },
}

for rel, reps in replacements.items():
    path = repo / rel
    if not path.exists():
        print(f"Atlandı, dosya yok: {path}")
        continue

    text = path.read_text()
    original = text

    for old, new in reps.items():
        text = text.replace(old, new)

    if text != original:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original)
        path.write_text(text)
        print(f"Patch uygulandı: {path}")
    else:
        print(f"Değişiklik gerekmedi: {path}")

print("Bitti.")
