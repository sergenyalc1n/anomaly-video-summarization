"""v1 text prototipleri.

13 anomali sinifi + 1 normal icin CLIP text embedding'leri uretir.
ONEMLI: Backbone, gorsel feature'lari cikardigin modelle BIREBIR ayni olmali
(src/clip_features.py -> open_clip ViT-B-16, laion2b_s34b_b88k). Yoksa text ve
gorsel vektorler ayni uzayda olmaz ve cosine similarity anlamsiz cikar.

Cikti: data/ucf/text_embeds_v1.npz
  - anomaly_protos : (13, 512) L2-normalize, ClassIDs.txt sirasinda
  - normal_proto   : (512,)    L2-normalize
  - class_names    : (13,) anomali sinif adlari

Kullanim:
  python tools/build_text_prototypes_v1.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import open_clip


# ClassIDs.txt'teki anomali siniflarinin dogal-dil karsiliklari.
# Normal_Videos_event burada YOK; o normal_proto tarafina gidiyor.
ANOMALY_CLASSES = {
    "Abuse": "abuse",
    "Arrest": "an arrest",
    "Arson": "arson, a deliberate fire",
    "Assault": "an assault",
    "Burglary": "a burglary",
    "Explosion": "an explosion",
    "Fighting": "people fighting",
    "RoadAccidents": "a road accident",
    "Robbery": "a robbery",
    "Shooting": "a shooting",
    "Shoplifting": "shoplifting",
    "Stealing": "stealing",
    "Vandalism": "vandalism",
}

# Prompt ensemble: her sinif icin bu sablonlarin ortalamasi alinir.
TEMPLATES = [
    "a surveillance video of {}",
    "a CCTV camera footage showing {}",
    "a security camera recording of {}",
    "footage of {} in a public place",
    "a video showing {}",
]

# Normal taraf icin ayri ensemble.
NORMAL_PHRASES = [
    "a normal everyday scene with no crime",
    "people doing ordinary daily activities",
    "a calm street with nothing unusual happening",
    "a normal surveillance video with no anomaly",
]


def build_prototypes(model_name, pretrained, device):
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode(prompts):
        toks = tokenizer(prompts).to(device)
        feats = model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalize
        proto = feats.mean(dim=0)                          # ensemble ortalamasi
        proto = proto / proto.norm()                       # ortalamayi tekrar normalize et
        return proto.float().cpu().numpy()

    names = list(ANOMALY_CLASSES.keys())
    anomaly_protos = np.stack(
        [encode([t.format(ANOMALY_CLASSES[n]) for t in TEMPLATES]) for n in names]
    ).astype(np.float32)  # (13, 512)

    normal_proto = encode(NORMAL_PHRASES).astype(np.float32)  # (512,)

    return anomaly_protos, normal_proto, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--out", default="data/ucf/text_embeds_v1.npz")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device} | backbone: {args.model_name} / {args.pretrained}")

    anomaly_protos, normal_proto, names = build_prototypes(
        args.model_name, args.pretrained, device
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        anomaly_protos=anomaly_protos,
        normal_proto=normal_proto,
        class_names=np.array(names),
    )

    # Hizli saglik kontrolu
    print(f"anomaly_protos: {anomaly_protos.shape}, normal_proto: {normal_proto.shape}")
    print(f"norm(anomaly[0])={np.linalg.norm(anomaly_protos[0]):.4f} (1.0 olmali)")
    print(f"norm(normal)={np.linalg.norm(normal_proto):.4f} (1.0 olmali)")
    print(f"Kaydedildi -> {out}")


if __name__ == "__main__":
    main()
