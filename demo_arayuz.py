"""
Anomali-Odaklı Video Özetleme — DEMO ARAYÜZÜ (Streamlit)

Kapak sayfası + canlı demo. Mevcut proje dosyalarını DEĞİŞTİRMEZ; sadece src modüllerini okur.

Çalıştırma:
    pip install streamlit pandas
    streamlit run demo_arayuz.py

YTÜ logosu için: çalışılan klasöre 'ytu_logo.png' (veya .jpg) koyun; otomatik gösterilir.
"""
import base64
import glob
import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from src.device import get_device
from src.feature_utils import process_feat
from src.anomaly_decision import has_anomaly
from src.summarizer import select_segments, segments_to_time_ranges
from src.video_io import get_video_duration, merge_time_ranges, write_summary_video

st.set_page_config(page_title="Anomali Özetleme — YTÜ", layout="wide", page_icon="🎓",
                   initial_sidebar_state="collapsed")

# ----------------------------- Tema / CSS -----------------------------
NAVY = "#102A54"
st.markdown(f"""
<style>
  .block-container {{ padding-top: 3.4rem; max-width: 1150px; }}
  #MainMenu, footer {{ visibility: hidden; }}
  h1, h2, h3 {{ color: {NAVY}; font-family: 'Segoe UI', Arial, sans-serif; }}
  .ytu-band {{ text-align:center; letter-spacing:3px; color:{NAVY}; font-weight:700; font-size:16px; }}
  .ytu-sub {{ text-align:center; color:{NAVY}; font-size:14px; margin-top:4px; font-weight:600; }}
  .ytu-title {{ text-align:center; color:{NAVY}; font-size:33px; font-weight:800; line-height:1.25; margin:24px 0 8px; }}
  .ytu-tag {{ text-align:center; color:{NAVY}; font-size:16px; margin-bottom:28px; font-weight:600; }}
  .ppl {{ display:flex; justify-content:center; gap:56px; flex-wrap:wrap; margin:8px 0 4px; }}
  .ppl .nm {{ color:{NAVY}; font-size:19px; font-weight:700; }}
  .ppl .no {{ color:{NAVY}; font-size:15px; font-weight:600; }}
  .advisor {{ text-align:center; color:{NAVY}; font-size:16px; margin-top:20px; font-weight:600; }}
  .advisor b {{ color:{NAVY}; font-weight:800; }}
  .stButton>button {{ background:{NAVY}; color:#fff; border:none; border-radius:8px;
     padding:0.6rem 1.6rem; font-size:16px; font-weight:600; }}
  .stButton>button:hover {{ background:#1c3f7a; color:#fff; }}
  .card {{ border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; background:#fafbfc; }}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "cover"


def find_logo():
    for ext in ("png", "jpg", "jpeg", "PNG", "JPG"):
        for name in (f"ytu_logo.{ext}", f"logo.{ext}", f"ytu.{ext}"):
            if os.path.exists(name):
                return name
    return None


# ============================== KAPAK ==============================
def show_cover():
    logo = find_logo()
    st.write("")
    if logo:
        ext = (Path(logo).suffix.lstrip(".").lower() or "png")
        if ext == "jpg":
            ext = "jpeg"
        b64 = base64.b64encode(open(logo, "rb").read()).decode()
        st.markdown(
            "<div style='text-align:center;'><div style='display:inline-block;background:#ffffff;"
            "border-radius:18px;padding:18px 24px;box-shadow:0 6px 22px rgba(0,0,0,0.30);'>"
            f"<img src='data:image/{ext};base64,{b64}' style='height:150px;display:block;'></div></div>",
            unsafe_allow_html=True)
        st.write("")
    else:
        st.markdown("<div style='text-align:center;color:#cdd6e6;font-size:13px;margin-top:8px;'>"
                    "(YTÜ logosu için klasöre <code>ytu_logo.png</code> ekleyin)</div>", unsafe_allow_html=True)

    st.markdown('<div class="ytu-band">YILDIZ TEKNİK ÜNİVERSİTESİ</div>', unsafe_allow_html=True)
    st.markdown('<div class="ytu-sub">Elektrik-Elektronik Fakültesi · Bilgisayar Mühendisliği Bölümü · Bitirme Projesi</div>', unsafe_allow_html=True)
    st.markdown('<div class="ytu-title">Güvenlik Kameralarından Alınan<br>Video Görüntülerinin Özetlenmesi</div>', unsafe_allow_html=True)
    st.markdown('<div class="ytu-tag">CLIP-TSA ve Metin-Görüntü Füzyonu ile Zayıf-Denetimli Anomali Özetleme</div>', unsafe_allow_html=True)

    st.markdown("""
      <div class="ppl">
        <div style="text-align:center;"><div class="nm">Ata Metin Türetken</div><div class="no">20001500</div></div>
        <div style="text-align:center;"><div class="nm">Sergen Yalçın</div><div class="no">23011507</div></div>
      </div>
      <div class="advisor">Proje Danışmanı: <b>Prof. Dr. Mine Elif Karslıgil</b></div>
    """, unsafe_allow_html=True)

    st.write("")
    c = st.columns([2, 1, 2])
    with c[1]:
        if st.button("Demoyu Başlat  ▶", use_container_width=True):
            st.session_state.page = "demo"
            st.rerun()


# ============================== DEMO ==============================
@st.cache_resource(show_spinner=False)
def load_scorer(ckpt, txt, a):
    device = get_device()
    from src.cliptsa_adapter_v1 import CLIPTSAScorerV1
    return CLIPTSAScorerV1(repo_dir="third_party/CLIP-TSA", checkpoint=ckpt,
                           text_embeds=txt, alpha=a, device=device), str(device)


@st.cache_data(show_spinner=False)
def build_feat_map(fd):
    return {Path(p).stem: p for p in glob.glob(f"{fd}/**/*.npy", recursive=True)}


def show_demo():
    # --- sabit (en iyi) model parametreleri ---
    alpha, top_k, nms_window = 0.3, 3, 1
    max_thr, topk_thr = 0.5472, 0.4576

    # üst şerit
    h = st.columns([6, 1])
    with h[0]:
        st.markdown(f"<h2 style='margin-bottom:0'>Anomali-Odaklı Video Özetleme</h2>"
                    f"<div style='color:{NAVY};font-weight:600'>v1 — Metin Füzyonlu Model · Yıldız Teknik Üniversitesi</div>",
                    unsafe_allow_html=True)
    with h[1]:
        if st.button("← Kapak"):
            st.session_state.page = "cover"; st.rerun()
    st.divider()

    # ayarlar (gizli genişletilebilir) — klasör + model yolları
    with st.expander("⚙️ Ayarlar — klasör ve model yolları"):
        video_dir = st.text_input("Video klasörü", "videos")
        feat_dir = st.text_input("Feature klasörü", "data/ucf/features/test")
        checkpoint = st.text_input("Model checkpoint (.pkl)", "checkpoints/v1_s1/cliptsa_ucf_v1_best.pkl")
        text_embeds = st.text_input("Metin prototipleri (.npz)", "data/ucf/text_embeds_v1.npz")

    if not os.path.exists(checkpoint):
        st.error(f"Checkpoint bulunamadı: `{checkpoint}` — Ayarlar bölümünden doğru yolu girin "
                 f"(ya da modeli eğitin).")
        return
    if not os.path.exists(text_embeds):
        st.error(f"Metin prototipleri bulunamadı: `{text_embeds}` — "
                 f"`python tools/build_text_prototypes_v1.py` ile üretin.")
        return

    feat_map = build_feat_map(feat_dir)
    vids = []
    for ext in ("*.mp4", "*.avi", "*.mkv"):
        vids += glob.glob(f"{video_dir}/**/{ext}", recursive=True)
    vids = sorted(v for v in vids if Path(v).stem in feat_map)

    if not vids:
        st.warning(f"`{video_dir}` altında, `{feat_dir}` feature'ı olan video bulunamadı. "
                   f"Ayarlar bölümünden klasörleri kontrol edin.")
        return

    sel_name = st.selectbox(f"📁 Test videosu seç  ({len(vids)} video)",
                            [Path(v).name for v in vids])
    video_path = next(v for v in vids if Path(v).name == sel_name)
    stem = Path(video_path).stem
    feat_path = feat_map.get(stem)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Orijinal video**")
        st.video(video_path)

    run = st.button("🔍 Analiz Et", type="primary")
    if not run:
        return

    try:
        with st.spinner("Model çalışıyor…"):
            scorer, dev = load_scorer(checkpoint, text_embeds, alpha)
            # Hard Attention perturbed top-k gürültüsünü sabitle -> her analiz aynı skor
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            raw = np.load(feat_path, allow_pickle=True).astype(np.float32)
            scores = np.asarray(scorer.score_features(process_feat(raw, 32, "mean")), dtype=np.float32)
            total = get_video_duration(video_path)
    except Exception as e:
        st.error("Analiz sırasında hata oluştu (ffmpeg kurulu mu?).")
        st.exception(e)
        return

    n = len(scores)
    anomaly, info = has_anomaly(scores, max_threshold=max_thr, topk_mean_threshold=topk_thr, k=3)

    with col2:
        st.markdown("**Karar**")
        if anomaly:
            st.markdown(f"<div class='card' style='border-color:#e11d48;background:#fff1f3;'>"
                        f"<span style='color:#e11d48;font-size:20px;font-weight:700;'>🚨 ANOMALİ BULUNDU</span><br>"
                        f"<span style='color:#555;'>max skor {info['max_score']:.3f} · top-3 ort. {info['topk_mean']:.3f}</span></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card' style='border-color:#16a34a;background:#f0fdf4;'>"
                        f"<span style='color:#16a34a;font-size:20px;font-weight:700;'>✅ NORMAL</span><br>"
                        f"<span style='color:#555;'>anomali eşiği aşılmadı (max {info['max_score']:.3f}) — özet üretilmez</span></div>",
                        unsafe_allow_html=True)

    if not anomaly:
        return

    selected = select_segments(scores, mode="peak", top_k=top_k, nms_window=nms_window)
    seg_dur = total / n if n else 0.0
    pre_sec, post_sec = 6.0, 2.0   # sonrası konsol ile aynı (2.0); yalnız ÖNCE 6 sn (konsol 2.0)
    ranges = merge_time_ranges([
        (max(0.0, i * seg_dur - pre_sec), min(total, (i + 1) * seg_dur + post_sec))
        for i in selected
    ])
    out_path = Path("outputs") / f"{stem}_demo_ozet.mp4"
    with st.spinner("Özet video oluşturuluyor (FFmpeg)…"):
        created = write_summary_video(video_path, ranges, out_path)

    st.divider()
    st.markdown("**📹 Anomali özeti**")
    if created and out_path.exists():
        c = st.columns([1, 2, 1])
        with c[1]:
            st.video(str(out_path))
    else:
        st.error("Özet video üretilemedi (FFmpeg kurulu mu?).")


# ----------------------------- Yönlendirme -----------------------------
if st.session_state.page == "cover":
    st.markdown(f"""<style>
      .stApp {{ background: linear-gradient(180deg, #0c1f42 0%, {NAVY} 55%, #16386f 100%); }}
      .ytu-band, .ytu-sub, .ytu-title, .ytu-tag,
      .ppl .nm, .ppl .no, .advisor, .advisor b {{ color:#FFFFFF !important; }}
      .stButton>button {{ background:#FFFFFF; color:{NAVY}; font-weight:700; }}
      .stButton>button:hover {{ background:#e8eefc; color:{NAVY}; }}
    </style>""", unsafe_allow_html=True)
    show_cover()
else:
    show_demo()
