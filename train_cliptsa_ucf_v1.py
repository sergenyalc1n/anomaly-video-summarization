"""v1 egitimi: CLIP-TSA + text-prototip fuzyonu (sabit alpha).

Egitim dongusu v0 (train_cliptsa_ucf.py) ile BIREBIR AYNI. Tek fark: model,
text fuzyonunu forward'in ICINE enjekte eden alt-sinif (src/text_fusion_v1).
Bu sayede alpha=0 -> bit-bit v0 (ayni seed -> ayni AUC), alpha>0 -> tum MIL
fused skor uzerinden calisir.

Guvenlik:
  - feature .npy dosyalarina YAZILMAZ (sadece okunur).
  - checkpoint'ler ayri klasore gider (checkpoints/ucf_v1), v0'i ezmez.

Kullanim:
  python train_cliptsa_ucf_v1.py --alpha 0.0            # = v0 (kontrol)
  python train_cliptsa_ucf_v1.py --alpha 0.3
  python train_cliptsa_ucf_v1.py --alpha 0.5
"""

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# v0 ile BIREBIR ayni davranis icin yardimcilari oradan import et
from train_cliptsa_ucf import (
    get_device,
    process_feat,
    UCFFeatureDataset,
    sparsity_loss,
    smooth_loss,
    pseudo_normal_loss,
    build_eval_index,
    expand_to_frames,
)

from src.text_fusion_v1 import build_v1_model


@torch.no_grad()
def evaluate_frame_auc(model, eval_items, device, num_segments=32, pool="mean"):
    """v0 ile ayni; out[6] artik fused skor (text forward icinde enjekte)."""
    model.eval()
    all_scores, all_labels = [], []
    for feat_path, spans, n_frames in eval_items:
        raw = np.load(feat_path, allow_pickle=True).astype(np.float32)  # SADECE OKUMA
        feats = process_feat(raw, length=num_segments, pool=pool)
        x = torch.from_numpy(feats[None, None, :, :]).float().to(device)
        out = model(x)
        seg = out[6].detach().float().cpu().reshape(-1).numpy()
        seg = seg[:num_segments]
        frame_scores = expand_to_frames(seg, n_frames)
        gt = np.zeros(n_frames, dtype=np.int32)
        for s, e in spans:
            if s >= 0 and e >= 0:
                gt[s:min(e + 1, n_frames)] = 1
        all_scores.append(frame_scores)
        all_labels.append(gt)
    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    return float(roc_auc_score(y_true, y_score))


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", default="data/ucf/features")
    parser.add_argument("--cliptsa_repo", default="third_party/CLIP-TSA")
    parser.add_argument("--output_dir", default="checkpoints/ucf_v1")  # v0'dan AYRI
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps_per_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_segments", type=int, default=32,
                        help="zamansal segment sayisi (16/32/64)")
    parser.add_argument("--disable_HA", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select_by", choices=["auc", "loss"], default="auc")
    parser.add_argument("--eval_annotation",
                        default="UCF_Crime/meta/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    parser.add_argument("--eval_frame_counts", default="data/ucf/features/frame_counts.json")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--pool", choices=["mean", "max"], default="mean")
    parser.add_argument("--sparsity_lambda", type=float, default=4e-2)
    parser.add_argument("--smooth_lambda", type=float, default=8e-4)
    parser.add_argument("--pseudo_normal_lambda", type=float, default=1.0)
    parser.add_argument("--pseudo_normal_k", type=int, default=16)
    # --- v1 text fuzyon parametreleri ---
    parser.add_argument("--text_embeds", default="data/ucf/text_embeds_v1.npz")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="fuzyon karisim katsayisi [0,1]. 0 = saf v0.")
    parser.add_argument("--tau", type=float, default=0.07,
                        help="text logit temperature (CLIP'te tipik 0.07)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Cihaz: {device} | alpha={args.alpha} tau={args.tau}")

    if not Path(args.text_embeds).exists():
        raise FileNotFoundError(
            f"text embeds bulunamadi: {args.text_embeds}\n"
            f"Once: python tools/build_text_prototypes_v1.py"
        )

    # --- v0 ile AYNI sira: datasets -> loaders -> model (RNG durumu eslessin) ---
    normal_ds = UCFFeatureDataset(args.feature_root, normal=True, pool=args.pool, num_segments=args.num_segments)
    abnormal_ds = UCFFeatureDataset(args.feature_root, normal=False, pool=args.pool, num_segments=args.num_segments)

    normal_loader = DataLoader(normal_ds, batch_size=args.batch_size, shuffle=True,
                               drop_last=True, num_workers=0)
    abnormal_loader = DataLoader(abnormal_ds, batch_size=args.batch_size, shuffle=True,
                                 drop_last=True, num_workers=0)

    model_args = SimpleNamespace(visual="vit", gpu="0", enable_HA=not args.disable_HA)
    model = build_v1_model(
        args.cliptsa_repo, 512, args.batch_size, args.k, args.num_samples,
        not args.disable_HA, model_args, args.text_embeds,
        alpha=args.alpha, tau=args.tau, device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005)
    bce = torch.nn.BCELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_items = None
    if args.select_by == "auc":
        if Path(args.eval_annotation).exists() and Path(args.eval_frame_counts).exists():
            eval_items = build_eval_index(args.feature_root, args.eval_annotation,
                                          args.eval_frame_counts, args.num_segments)
        if not eval_items:
            print("UYARI: AUC dosyalari yok; loss tabanli secime donuluyor.")
            args.select_by = "loss"

    best_loss = float("inf")
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        n_iter = iter(normal_loader)
        a_iter = iter(abnormal_loader)
        pbar = tqdm(range(args.steps_per_epoch), desc=f"v1 Epoch {epoch}/{args.epochs}")
        epoch_losses = []

        for _ in pbar:
            try:
                ninput, nlabel = next(n_iter)
            except StopIteration:
                n_iter = iter(normal_loader); ninput, nlabel = next(n_iter)
            try:
                ainput, alabel = next(a_iter)
            except StopIteration:
                a_iter = iter(abnormal_loader); ainput, alabel = next(a_iter)

            inputs = torch.cat([ninput, ainput], dim=0).to(device).float()

            # --- v0 ile BIREBIR ayni loss ---
            outputs = model(inputs)
            score_abnormal, score_normal = outputs[0], outputs[1]
            all_scores = outputs[6]

            score_normal = score_normal.squeeze()
            score_abnormal = score_abnormal.squeeze()

            loss_cls = bce(score_normal, torch.zeros_like(score_normal)) + \
                       bce(score_abnormal, torch.ones_like(score_abnormal))

            abnormal_segment_scores = all_scores[args.batch_size:].squeeze(-1)
            loss_sp = sparsity_loss(abnormal_segment_scores, lam=args.sparsity_lambda)
            loss_sm = smooth_loss(abnormal_segment_scores, lam=args.smooth_lambda)
            loss_pn = pseudo_normal_loss(abnormal_segment_scores,
                                         k_bottom=args.pseudo_normal_k,
                                         lam=args.pseudo_normal_lambda)

            loss = loss_cls + loss_sp + loss_sm + loss_pn

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            abn_seg_mean = abnormal_segment_scores.mean().item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_cls.item():.3f}",
                "pn": f"{loss_pn.item():.3f}",
                "abn_top3": f"{score_abnormal.mean().item():.3f}",
                "abn_all": f"{abn_seg_mean:.3f}",
            })

        avg_loss = float(np.mean(epoch_losses))
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        epoch_auc = None
        if args.select_by == "auc" and eval_items and (epoch % args.eval_every == 0):
            epoch_auc = evaluate_frame_auc(model, eval_items, device,
                                           args.num_segments, pool=args.pool)
            print(f"Epoch {epoch} frame-level AUC (fused): {epoch_auc:.4f}")
            model.train()

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "avg_loss": avg_loss,
            "frame_auc": epoch_auc,
            "args": vars(args),
        }
        torch.save(ckpt, output_dir / "cliptsa_ucf_v1_last.pkl")

        if args.select_by == "auc":
            if epoch_auc is not None and epoch_auc > best_auc:
                best_auc = epoch_auc
                torch.save(ckpt, output_dir / "cliptsa_ucf_v1_best.pkl")
                print(f"Yeni en iyi (AUC={best_auc:.4f}) -> {output_dir / 'cliptsa_ucf_v1_best.pkl'}")
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, output_dir / "cliptsa_ucf_v1_best.pkl")
                print(f"Yeni en iyi (loss={best_loss:.4f}) -> {output_dir / 'cliptsa_ucf_v1_best.pkl'}")

    print("v1 egitim bitti.")
    print(f"Son: {output_dir / 'cliptsa_ucf_v1_last.pkl'}")
    print(f"En iyi: {output_dir / 'cliptsa_ucf_v1_best.pkl'}")
    if args.select_by == "auc" and best_auc >= 0:
        print(f"En iyi frame-level AUC (fused): {best_auc:.4f}")


if __name__ == "__main__":
    train()
