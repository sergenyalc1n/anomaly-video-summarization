"""v1: CLIP-TSA + text-prototip fuzyonu (modelin ICINE enjekte edilmis).

Onceki tasarim egitim dongusunu yeniden yaziyordu -> v0 ile birebir olmuyordu
(RNG akisi + MIL detaylari). Bu surum bunun yerine CLIP-TSA Model'i SUBCLASS edip
forward'i BIREBIR kopyalar ve sadece TEK satir ekler: segment skorlari sigmoid'den
ciktiktan sonra text-similarity ile harmanlanir.

  scores = (1 - alpha) * gorsel + alpha * text_prob      # (bs, T)

alpha=0  -> ek satir hic calismaz (guard) -> forward BIREBIR v0 ile ayni
            -> ayni seed'de bit-bit ayni egitim ve AUC.
alpha>0  -> tum MIL makinesi (magnitude top-k secimi, dropout, loss'lar)
            fused skor uzerinden calisir; egitim dongusu v0'inkiyle AYNI kalir.

build_v1_model(...) bir Model alt-sinifi orneklendirip dondurur. CLIP-TSA repo'su
sys.path'e burada eklenir, boylece train scripti sadece bu fonksiyonu cagirir.
"""

import sys
from pathlib import Path

import numpy as np
import torch


def build_v1_model(repo_dir, feature_size, batch_size, k, num_samples,
                   enable_HA, args, text_embeds_path, alpha=0.5, tau=0.07,
                   device=None):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from model import Model  # CLIP-TSA orijinal modeli

    data = np.load(text_embeds_path, allow_pickle=True)
    anomaly_np = data["anomaly_protos"].astype(np.float32)  # (C, F)
    normal_np = data["normal_proto"].astype(np.float32)     # (F,)

    class CLIPTSATextFusion(Model):
        def __init__(self):
            super().__init__(feature_size, batch_size, k, num_samples, enable_HA, args)
            self.alpha = float(alpha)
            self.tau = float(tau)
            self.register_buffer("anomaly_protos", torch.from_numpy(anomaly_np))
            self.register_buffer("normal_proto", torch.from_numpy(normal_np))

        def _text_prob(self, inputs):
            """inputs: (bs, ncrops, t, f) -> text_prob: (bs, t) in [0,1]."""
            seg = inputs.mean(dim=1)                                   # (bs, t, f)
            seg = seg / (seg.norm(dim=-1, keepdim=True) + 1e-8)
            sim_anom = torch.einsum("btf,cf->btc", seg, self.anomaly_protos)
            sim_anom = sim_anom.max(dim=-1).values                    # (bs, t)
            sim_norm = torch.einsum("btf,f->bt", seg, self.normal_proto)
            return torch.sigmoid((sim_anom - sim_norm) / self.tau)

        # --- v0 model.py forward'inin BIREBIR kopyasi + tek enjeksiyon satiri ---
        def forward(self, inputs):
            k_abn = self.k_abn
            k_nor = self.k_nor

            out = inputs
            bs, ncrops, t, f = out.size()
            out = out.view(-1, t, f)

            if f > 512:
                out = self.mlp(out)
                f = 512

            if self.apply_HA:
                if self.visual != "vit" and out.shape[0] > 1 and out.shape[1] != 32:
                    concat = []
                    for i in out:
                        concat.append(self.hard_attention(i.unsqueeze(0)))
                    out = torch.cat(concat, dim=0)
                else:
                    out = self.hard_attention(out)

            out = self.Aggregate(out)
            out = self.drop_out(out)

            features = out
            scores = self.relu(self.fc1(features))
            scores = self.drop_out(scores)
            scores = self.relu(self.fc2(scores))
            scores = self.drop_out(scores)
            scores = self.sigmoid(self.fc3(scores))
            scores = scores.view(bs, ncrops, -1).mean(1)   # (bs, t)

            # >>> TEXT FUZYONU (tek eklenen satir) <<<
            # alpha=0 ise guard sayesinde hic dokunulmaz -> birebir v0.
            if self.alpha != 0.0:
                scores = (1.0 - self.alpha) * scores + self.alpha * self._text_prob(inputs)

            scores = scores.unsqueeze(dim=2)               # (bs, t, 1)

            adjusted_scoremag_batch_size = int(self.batch_size * self.parallel)
            adjusted_feat_batch_size = int(self.batch_size * ncrops * self.parallel)

            normal_features = features[0:adjusted_feat_batch_size]
            normal_scores = scores[0:adjusted_scoremag_batch_size]

            abnormal_features = features[adjusted_feat_batch_size:]
            abnormal_scores = scores[adjusted_scoremag_batch_size:]

            feat_magnitudes = torch.norm(features, p=2, dim=2)
            feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
            nfea_magnitudes = feat_magnitudes[0:adjusted_scoremag_batch_size]
            afea_magnitudes = feat_magnitudes[adjusted_scoremag_batch_size:]
            n_size = nfea_magnitudes.shape[0]

            if nfea_magnitudes.shape[0] == 1:  # inference, batch size 1
                afea_magnitudes = nfea_magnitudes
                abnormal_scores = normal_scores
                abnormal_features = normal_features

            # abnormal: top-k feature magnitude
            select_idx = torch.ones_like(nfea_magnitudes, device=nfea_magnitudes.device)
            select_idx = self.drop_out(select_idx)
            afea_magnitudes_drop = afea_magnitudes * select_idx
            idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]

            idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
            abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
            abnormal_features = abnormal_features.permute(1, 0, 2, 3)

            total_select_abn_feature = torch.zeros(0, device=features.device)
            for abnormal_feature in abnormal_features:
                feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)
                total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

            idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
            score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)

            # normal: top-k feature magnitude
            select_idx_normal = torch.ones_like(nfea_magnitudes, device=nfea_magnitudes.device)
            select_idx_normal = self.drop_out(select_idx_normal)
            nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
            idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]

            idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])
            normal_features = normal_features.view(n_size, ncrops, t, f)
            normal_features = normal_features.permute(1, 0, 2, 3)

            total_select_nor_feature = torch.zeros(0, device=features.device)
            for nor_fea in normal_features:
                feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)
                total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

            idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
            score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

            feat_select_abn = total_select_abn_feature
            feat_select_normal = total_select_nor_feature

            return (score_abnormal, score_normal, feat_select_abn, feat_select_normal,
                    feat_select_abn, feat_select_abn, scores, feat_select_abn,
                    feat_select_abn, feat_magnitudes)

    model = CLIPTSATextFusion()
    if device is not None:
        model = model.to(device)
    return model
