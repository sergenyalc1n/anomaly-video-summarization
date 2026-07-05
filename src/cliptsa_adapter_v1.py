"""v1 (text-fuzyon) inference adaptoru.

Temel CLIPTSAScorer (src/cliptsa_adapter.py) DUZ gorsel modeli yukler; text
fuzyonu yoktur. Bu surum bunun yerine v1 modelini (src/text_fusion_v1.build_v1_model)
orneklendirir: CLIP-TSA + DONUK text-prototip fuzyonu (sabit alpha). Boylece
inference'ta uretilen segment skorlari (out[6]) EGITIMDEKIYLE ayni fused skordur.

Inference'ta CLIP text encoder GEREKMEZ: v1 prototipleri onceden hesaplanmis
.npz buffer'dir (data/ucf/text_embeds_v1.npz). Yani hafif ve hizli.

Skor cikarma / girdi hazirlama mantigi temel scorer ile AYNI; sadece __init__
farkli (v1 modeli + alpha/tau/text_embeds).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from src.cliptsa_adapter import CLIPTSAScorer


class CLIPTSAScorerV1(CLIPTSAScorer):
    def __init__(
        self,
        repo_dir,
        checkpoint,
        text_embeds="data/ucf/text_embeds_v1.npz",
        feature_size=512,
        k=0.95,
        num_samples=32,
        enable_HA=True,
        alpha=0.3,
        tau=0.07,
        device=None,
    ):
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.text_embeds = Path(text_embeds).expanduser().resolve()
        self.feature_size = int(feature_size)
        self.k = float(k)
        self.num_samples = int(num_samples)
        self.enable_HA = bool(enable_HA)
        self.alpha = float(alpha)
        self.tau = float(tau)

        if not self.repo_dir.exists():
            raise FileNotFoundError(f"CLIP-TSA repo klasoru bulunamadi: {self.repo_dir}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint bulunamadi: {self.checkpoint}")
        if not self.text_embeds.exists():
            raise FileNotFoundError(
                f"Text prototipleri bulunamadi: {self.text_embeds}\n"
                f"Once: python tools/build_text_prototypes_v1.py"
            )

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # v1 modelini kur (build_v1_model RNG'ye dokunmaz; sadece insa eder).
        from src.text_fusion_v1 import build_v1_model

        model_args = SimpleNamespace(gpu="0", enable_HA=self.enable_HA, visual="vit")
        # batch_size=1 -> v1 forward'in "inference, batch size 1" guard'i devreye girer.
        self.model = build_v1_model(
            str(self.repo_dir), self.feature_size, 1, self.k, self.num_samples,
            self.enable_HA, model_args, str(self.text_embeds),
            alpha=self.alpha, tau=self.tau, device=self.device,
        )

        ckpt = torch.load(str(self.checkpoint), map_location="cpu")
        state = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt

        clean_state = {}
        for key, value in state.items():
            clean_state[key[len("module."):] if key.startswith("module.") else key] = value

        # strict=False: donuk prototip buffer'lari zaten build sirasinda yuklendi;
        # checkpoint'teki ayni degerleri tekrar set etmek sorun degil.
        self.model.load_state_dict(clean_state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"v1 scorer hazir | alpha={self.alpha} tau={self.tau} | "
              f"checkpoint={self.checkpoint.name}")

    # _prepare_input, _extract_segment_scores, score_features -> temel siniftan miras.
