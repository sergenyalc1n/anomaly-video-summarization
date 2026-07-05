import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class CLIPTSAScorer:
    """
    Gerçek CLIP-TSA checkpoint'i ile segment-level anomaly score üretir.
    Geçici/heuristic skor üretmez.
    """

    def __init__(
        self,
        repo_dir,
        checkpoint,
        feature_size=512,
        k=0.95,
        num_samples=32,
        enable_HA=True,
        device=None,
    ):
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.feature_size = int(feature_size)
        self.k = float(k)
        self.num_samples = int(num_samples)
        self.enable_HA = bool(enable_HA)

        if not self.repo_dir.exists():
            raise FileNotFoundError(f"CLIP-TSA repo klasörü bulunamadı: {self.repo_dir}")

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint bulunamadı: {self.checkpoint}")

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        from model import Model

        self.device = device or (
            # Mac (MPS) code commented out for Windows
            # torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # CLIP-TSA model.py args.gpu ve args.enable_HA bekliyor.
        args = SimpleNamespace(
            gpu="0",
            enable_HA=self.enable_HA,
            visual="vit",
        )

        self.model = Model(
            self.feature_size,
            1,
            self.k,
            self.num_samples,
            self.enable_HA,
            args,
        )

        ckpt = torch.load(str(self.checkpoint), map_location="cpu")

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        clean_state = {}
        for key, value in state.items():
            if key.startswith("module."):
                clean_state[key[len("module."):]] = value
            else:
                clean_state[key] = value

        self.model.load_state_dict(clean_state, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_input(self, features):
        x = np.asarray(features, dtype=np.float32)

        if x.ndim == 2:
            # (T, F) -> (1, 1, T, F)
            x = x[None, None, :, :]
        elif x.ndim == 3:
            # (C, T, F) -> (1, C, T, F)
            x = x[None, :, :, :]
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Desteklenmeyen feature shape: {x.shape}")

        return torch.from_numpy(x).float().to(self.device)

    def _extract_segment_scores(self, output, expected_segments):
        if isinstance(output, (tuple, list)):
            # Resmi CLIP-TSA modelinde segment skorları genelde outputs[6].
            if len(output) > 6 and torch.is_tensor(output[6]):
                y = output[6]
            else:
                candidates = [o for o in output if torch.is_tensor(o)]
                matching = [o for o in candidates if expected_segments in list(o.shape)]
                if matching:
                    y = matching[-1]
                else:
                    raise RuntimeError("CLIP-TSA output içinde segment skorları bulunamadı.")
        elif torch.is_tensor(output):
            y = output
        else:
            raise RuntimeError(f"Model output desteklenmiyor: {type(output)}")

        y = y.detach().float().cpu()

        if y.min() < 0 or y.max() > 1:
            y = torch.sigmoid(y)

        y = y.reshape(-1).numpy().astype(np.float32)

        if len(y) < expected_segments:
            raise RuntimeError(
                f"Skor sayısı az. Beklenen={expected_segments}, gelen={len(y)}"
            )

        return y[:expected_segments]

    def score_features(self, features):
        features = np.asarray(features, dtype=np.float32)
        expected_segments = features.shape[-2]

        x = self._prepare_input(features)

        with torch.no_grad():
            output = self.model(x)

        return self._extract_segment_scores(output, expected_segments)
