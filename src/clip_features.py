import cv2
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


def load_clip(device):
    # NOT: Bu backbone, colab_extract_features.ipynb ile BIREBIR ayni olmali.
    # Egitim feature'lari hangi modelle cikarildiysa inference da onu kullanmali
    # (yoksa distribution shift olur). Su an: ViT-B/16.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="laion2b_s34b_b88k",
    )
    model = model.to(device).eval()
    return model, preprocess


def extract_clip_features(video_path, device, sample_fps=2):
    model, preprocess = load_clip(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(round(original_fps / sample_fps)), 1)

    features = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="CLIP feature çıkarılıyor")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                feat = model.encode_image(image_tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)

                features.append(feat.squeeze(0).detach().cpu().numpy())

            frame_idx += 1
            pbar.update(1)

    pbar.close()
    cap.release()

    if len(features) == 0:
        raise RuntimeError("Videodan feature çıkarılamadı.")

    return np.asarray(features, dtype=np.float32)


def temporal_segment_features(frame_features, num_segments=32):
    frame_features = np.asarray(frame_features, dtype=np.float32)
    n = len(frame_features)

    if n == 0:
        raise ValueError("Boş feature dizisi.")

    edges = np.linspace(0, n, num_segments + 1, dtype=int)
    segments = []

    for i in range(num_segments):
        start, end = edges[i], edges[i + 1]

        if end <= start:
            idx = min(start, n - 1)
            seg = frame_features[idx]
        else:
            seg = frame_features[start:end].mean(axis=0)

        segments.append(seg)

    return np.asarray(segments, dtype=np.float32)
