"""
app.py — Phân Loại Cây Xanh Phường Bến Nghé
=========================================
Method 1 : ConvNeXtV2-Tiny  (single model)
Method 2 : Cascaded Ensemble (ConvNeXt + FastViT + MobileNetV4 + DINOv2)
"""

import json
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# ╔══════════════════════════════════════════════════════╗
#   CONSTANTS
# ╚══════════════════════════════════════════════════════╝
INPUT_SIZE     = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]
INFO_JSON_PATH = "info19.json"
NUM_CLASSES    = 19

CLASS_NAMES = [
    "BoDe", "CamXe", "DauRai", "GiaTy", "GiangHuong",
    "GoMat", "KenHong", "LimSet", "LocVung", "MacNua",
    "MeTay", "MeXanh", "NhacNgua", "SaoDen", "Si",
    "SoKhi", "SuDai", "Trai", "Viet",
]

CLASS_NAMES_VN = {
    "BoDe"      : "Bồ Đề",
    "CamXe"     : "Cẩm Xe",
    "DauRai"    : "Dầu Rái",
    "GiaTy"     : "Giá Tị",
    "GiangHuong": "Giáng Hương",
    "GoMat"     : "Gõ Mật",
    "KenHong"   : "Kèn Hồng",
    "LimSet"    : "Lim Sẹt",
    "LocVung"   : "Lộc Vừng",
    "MacNua"    : "Mắc Nưa",
    "MeTay"     : "Mè Tây",
    "MeXanh"    : "Me Xanh",
    "NhacNgua"  : "Nhạc Ngựa",
    "SaoDen"    : "Sao Đen",
    "Si"        : "Si",
    "SoKhi"     : "Sồi Khỉ",
    "SuDai"     : "Sứ Đại",
    "Trai"      : "Trái",
    "Viet"      : "Việt",
}

# ── Model paths (flat layout) ──────────────────────────
CONV_PATH    = "ConvNeXtV2Tiny_S1_CrossEntropy.pth"  # ConvNeXt đơn
FUS_CONVNEXT = "ConvNeXtV2Tiny_S1_CrossEntropy.pth"  # Fusion backbone
FUS_FASTVIT  = "FastViT_S1_CrossEntropy.pth"
FUS_HEAD     = "fusion_best.pth"
ENS_CONVNEXT = "ConvNeXtV2Tiny_S1_CrossEntropy.pth"  # Ensemble
ENS_FASTVIT  = "FastViT_S1_CrossEntropy.pth"
ENS_MOBILE   = "best_model.pth"
ENS_DINO     = "DINOv2_ViTS14_LPFT_Final_ema.pth"

# ╔══════════════════════════════════════════════════════╗
#   PAGE CONFIG & CSS
# ╚══════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="Phân Loại Cây Xanh",
    page_icon="🌿",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Lora:ital,wght@0,600;1,400&display=swap');
:root {
    --green-dark  : #1a3c2e;
    --green-mid   : #2d6a4f;
    --green-light : #52b788;
    --green-pale  : #d8f3dc;
    --bg          : #f8fdf9;
    --card        : #ffffff;
    --text        : #1b2d24;
    --muted       : #6b8f71;
    --radius      : 14px;
    --shadow      : 0 4px 24px rgba(30,80,50,.10);
}
/* Ẩn HuggingFace Spaces injected navbar */
iframe { border: none !important; }
body > div:first-child:not([id]) { display: none !important; }

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background: var(--bg);
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]       { display: none !important; }
[data-testid="stDecoration"]    { display: none !important; }
[data-testid="stStatusWidget"]  { display: none !important; }
/* Ẩn bất kỳ text_input nào render ngoài ý muốn ở đầu trang */
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 760px; }
.page-subtitle { font-size: .93rem; color: var(--muted); margin-bottom: 1.6rem; }
.method-box {
    background: var(--card);
    border: 1px solid #e4f0e8;
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
}
.method-title { font-weight: 800; color: var(--green-dark); margin-bottom: .5rem; font-size: .95rem; }
.stFileUploader > div {
    border: 2px dashed var(--green-light) !important;
    border-radius: var(--radius) !important;
    background: var(--green-pale) !important;
}
.stFileUploader > div:hover { border-color: var(--green-mid) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid var(--green-pale); }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: .5rem 1.4rem !important;
    font-weight: 700 !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--green-pale) !important;
    color: var(--green-dark) !important;
    border-bottom: 3px solid var(--green-mid) !important;
}
/* ── Ẩn hoàn toàn label radio ── */
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] > div:first-child { display: none !important; }
div[data-testid="stRadio"] > div > label {
    background: var(--green-pale);
    border-radius: 10px;
    padding: .45rem 1.1rem;
    font-weight: 700;
    color: var(--green-dark);
    transition: background .15s;
    cursor: pointer;
}
div[data-testid="stRadio"] > div > label:hover { background: #b7e4c7; }
div[data-testid="stRadio"] > div { gap: 0.8rem; flex-wrap: wrap; }
/* ── Nút bấm ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--green-mid), var(--green-light));
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: .6rem 1.6rem !important;
    box-shadow: 0 3px 12px rgba(45,106,79,.30);
    transition: opacity .2s, transform .1s;
}
.stButton > button:hover  { opacity: .88; transform: translateY(-1px); }
.stButton > button:active { opacity: 1;   transform: translateY(0);    }
.result-banner {
    background: linear-gradient(135deg, var(--green-dark), var(--green-mid));
    color: #fff; border-radius: var(--radius);
    padding: 1.4rem; text-align: center; margin-bottom: 1rem;
}
.result-banner .tree-name  { font-family: 'Lora', serif; font-size: 1.8rem; font-weight: 600; }
.result-banner .confidence { font-size: .95rem; opacity: .85; margin-top: .3rem; }
.result-banner .method-tag { font-size: .78rem; opacity: .70; margin-top: .25rem; }
.result-banner .demo-note  { font-size: .75rem; opacity: .65; margin-top: .35rem; }
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .7rem; margin-top: .8rem; }
.chip      { background: var(--green-pale); border-radius: 10px; padding: .6rem .9rem; }
.chip .lbl { color: var(--muted); font-size: .72rem; font-weight: 700;
             text-transform: uppercase; letter-spacing: .05em; }
.chip .val { color: var(--green-dark); font-weight: 700; margin-top: .15rem; font-size: .83rem; }
.badge        { display:inline-block; padding:.2rem .7rem; border-radius:99px; font-size:.78rem; font-weight:700; }
.badge-green  { background:#d8f3dc; color:var(--green-dark); }
.badge-yellow { background:#fff3cd; color:#7d5a00; }
.badge-red    { background:#fde8e8; color:#9b1c1c; }
.placeholder {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 200px;
    background: var(--green-pale); border-radius: var(--radius);
    color: var(--muted); font-weight: 600; gap: .5rem;
}
.topk-row { display:flex; align-items:center; gap:.6rem; margin:.3rem 0; }
.topk-label { min-width:120px; font-weight:700; font-size:.82rem; color:var(--green-dark); }
.topk-bar-bg { flex:1; background:#e4f0e8; border-radius:6px; height:10px; overflow:hidden; }
.topk-bar { height:100%; background:var(--green-light); border-radius:6px; }
.topk-pct { font-size:.82rem; font-weight:700; color:var(--muted); min-width:42px; text-align:right; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════╗
#   SESSION STATE
# ╚══════════════════════════════════════════════════════╝
if "result"      not in st.session_state: st.session_state.result = None
if "method_used" not in st.session_state: st.session_state.method_used = None

# ╔══════════════════════════════════════════════════════╗
#   TRANSFORM
# ╚══════════════════════════════════════════════════════╝
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

# ╔══════════════════════════════════════════════════════╗
#   MODEL LOADERS
# ╚══════════════════════════════════════════════════════╝
@st.cache_resource(show_spinner=False)
def load_conv(path: str):
    """Load ConvNeXtV2-Tiny đơn."""
    try:
        import timm
        model = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=False, num_classes=NUM_CLASSES,
        )
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model, None
    except FileNotFoundError:
        return None, f"Không tìm thấy: {path}"
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_fusion(convnext_path, fastvit_path, fusion_path):
    """Load Fusion (ConvNeXt + FastViT + WeightedFusionClassifier)."""
    try:
        from fusion_model import load_fusion_models
        models = load_fusion_models(
            convnext_path=convnext_path,
            fastvit_path=fastvit_path,
            fusion_path=fusion_path,
            device=torch.device("cpu"),
        )
        return models, None
    except FileNotFoundError as e:
        return None, f"Không tìm thấy: {e}"
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_ens(convnext_path, fastvit_path, mobile_path, dino_path):
    """Load Cascaded Ensemble 4 model."""
    try:
        from ensemble_model import load_ensemble as _load
        models = _load(
            convnext_path=convnext_path,
            fastvit_path=fastvit_path,
            mobilev4_path=mobile_path,
            dinov2_path=dino_path,
            device=torch.device("cpu"),
        )
        return models, None
    except FileNotFoundError as e:
        return None, f"Không tìm thấy: {e}"
    except Exception as e:
        return None, str(e)


# ╔══════════════════════════════════════════════════════╗
#   INFERENCE
# ╚══════════════════════════════════════════════════════╝
def _dummy_result(label: str) -> dict:
    scores = np.random.dirichlet(np.ones(NUM_CLASSES) * 0.3)
    idx    = scores.argmax()
    return {
        "top1_class"      : CLASS_NAMES_VN.get(CLASS_NAMES[idx], CLASS_NAMES[idx]),
        "top1_confidence" : float(scores[idx]),
        "is_dummy"        : True,
        "method"          : label,
    }


def predict_conv(model, image: Image.Image) -> dict:
    label = "ConvNeXtV2-Tiny"
    if model is None:
        return _dummy_result(label)
    tensor = IMG_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()
    idx = int(probs.argmax())
    return {
        "top1_class"      : CLASS_NAMES_VN.get(CLASS_NAMES[idx], CLASS_NAMES[idx]),
        "top1_confidence" : float(probs[idx]),
        "is_dummy"        : False,
        "method"          : label,
    }


def predict_fusion(fusion_models: dict, image: Image.Image) -> dict:
    label = "Fusion (ConvNeXtV2 + FastViT)"
    if fusion_models is None:
        return _dummy_result(label)
    try:
        from fusion_model import predict_fusion as _fus
        vn_names = list(CLASS_NAMES_VN.values())
        res = _fus(fusion_models, image, class_names=vn_names, top_k=1)
        return {
            "top1_class"      : res["predicted_class"],
            "top1_confidence" : res["confidence"],
            "is_dummy"        : False,
            "method"          : label,
        }
    except Exception as e:
        st.warning(f"⚠️ Fusion error: {e}")
        return _dummy_result(label)


def predict_ensemble(ens_models: dict, image: Image.Image) -> dict:
    label = "Cascaded Ensemble (ConvNeXt + FastViT + MobileNetV4 + DINOv2)"
    if ens_models is None:
        return _dummy_result(label)
    try:
        from ensemble_model import predict as _ens
        res = _ens(ens_models, image, use_tta=False, top_k=1)
        cls       = CLASS_NAMES_VN.get(res["predicted_class"], res["predicted_class"])
        path_note = f" [{res.get('path','full')} path]"
        return {
            "top1_class"      : cls,
            "top1_confidence" : float(res["confidence"]),
            "is_dummy"        : False,
            "method"          : label + path_note,
        }
    except Exception as e:
        st.warning(f"⚠️ Ensemble error: {e}")
        return _dummy_result(label)


# ╔══════════════════════════════════════════════════════╗
#   LOAD INFO JSON
# ╚══════════════════════════════════════════════════════╝
@st.cache_data(show_spinner=False)
def load_info() -> dict:
    try:
        with open(INFO_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ╔══════════════════════════════════════════════════════╗
#   HELPER UI
# ╚══════════════════════════════════════════════════════╝
def conservation_badge(status: str) -> str:
    s = status.lower()
    if any(k in s for k in ["least concern", "lc", "an toàn"]):
        css = "badge-green"
    elif any(k in s for k in ["vulnerable", "vu", "dễ bị"]):
        css = "badge-yellow"
    else:
        css = "badge-red"
    return f'<span class="badge {css}">{status}</span>'


def render_tree_info(class_name: str, info_db: dict):
    info = info_db.get(class_name)
    if not info:
        st.info("Chưa có thông tin chi tiết cho loài này.")
        return
    img_col, desc_col = st.columns([1, 2])
    with img_col:
        if info.get("image_url"):
            st.image(info["image_url"], use_container_width=True,
                     caption=info.get("scientific_name", ""))
    with desc_col:
        sci = info.get("scientific_name", "—")
        st.markdown(f"**Tên khoa học:** *{sci}*")
        st.markdown(f"**🌿 Họ thực vật:** {info.get('family', '—')}")
        common = ", ".join(info.get("common_names", [])) or "—"
        st.markdown(f"**Tên thường gọi:** {common}")
        st.markdown(info.get("description", ""))
    st.markdown("---")
    h = info.get("height_range_m", ["?", "?"])
    c = info.get("canopy_spread_m", ["?", "?"])
    chips = [
        ("Chiều cao",   f"{h[0]}–{h[-1]} m"),
        ("Tán cây",     f"{c[0]}–{c[-1]} m"),
        ("Tốc độ tăng", info.get("growth_rate",    "—")),
        ("Tuổi thọ",    info.get("lifespan_years",  "—")),
        ("Bộ rễ",       info.get("root_system",     "—")),
        ("Lá",          info.get("leaf_description","—")),
        ("Vỏ cây",      info.get("bark",            "—")),
        ("Quả",         info.get("fruit",           "—")),
    ]
    grid = "".join(
        f'<div class="chip"><div class="lbl">{lbl}</div>'
        f'<div class="val">{val}</div></div>'
        for lbl, val in chips
    )
    st.markdown(f'<div class="info-grid">{grid}</div>', unsafe_allow_html=True)
    st.markdown("")
    cs = info.get("conservation_status", "—")
    st.markdown(f"**Tình trạng hiện nay:** {conservation_badge(cs)}", unsafe_allow_html=True)
    uses = info.get("uses", [])
    if uses:
        st.markdown(f"**Công dụng cây:** {', '.join(uses)}")
    if info.get("fun_fact"):
        st.info(f"💡 {info['fun_fact']}")
    dist = info.get("distribution", [])
    if dist:
        st.markdown(f"**Phân bố:** {', '.join(dist)}")
    if info.get("wikipedia_url"):
        st.markdown(f"[Xem thêm trên Wikipedia]({info['wikipedia_url']})")


# ╔══════════════════════════════════════════════════════╗
#   INITIALIZATION
# ╚══════════════════════════════════════════════════════╝
info_db = load_info()

# ╔══════════════════════════════════════════════════════╗
#   APPLICATION
# ╚══════════════════════════════════════════════════════╝
st.markdown(
    '<p class="page-subtitle">ỨNG DỤNG KỸ THUẬT DEEP LEARNING ĐỂ PHÂN LOẠI CÂY XANH '
    'PHƯỜNG BẾN NGHÉ VÀ XÂY DỰNG ỨNG DỤNG</p>',
    unsafe_allow_html=True,
)

# ── Input + method selector ────────────────────────────
if not st.session_state.result:

    # ── Method selector (radio — chỉ chọn 1) ──────────
    st.markdown('''<div class="method-box">
<div class="method-title">🔬 Chọn phương pháp phân loại</div>''', unsafe_allow_html=True)
    method_choice = st.radio(
        label=" ",
        options=["ConvNeXtV2-Tiny", "Fusion", "Cascaded Ensemble"],
        index=0,
        horizontal=True,
        label_visibility="hidden",
    )
    captions = {
        "ConvNeXtV2-Tiny"   : "⚙️ **ConvNeXtV2-Tiny:** Mô hình đơn — nhanh, nhẹ.",
        "Fusion"            : "⚙️ **Fusion:** ConvNeXtV2-Tiny + FastViT-S12 kết hợp qua Weighted Fusion Classifier.",
        "Cascaded Ensemble" : "⚙️ **Cascaded Ensemble:** 4 model (ConvNeXt + FastViT + MobileNetV4 + DINOv2), gate tự động.",
    }
    st.markdown(f'<p style="font-size:.88rem;color:#6b8f71;margin-top:.2rem">{captions[method_choice]}</p>',
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Input ảnh ──────────────────────────────────────
    st.markdown("#### Tải ảnh hoặc chụp hình nếu sử dụng điện thoại.")
    tab_upload, tab_camera = st.tabs(["Tải ảnh", "Chụp ảnh"])
    input_image = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Chọn ảnh", type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            help="Chỉ chọn 1 ảnh duy nhất",
        )
        if uploaded is not None:
            input_image = Image.open(uploaded).convert("RGB")

    with tab_camera:
        photo = st.camera_input("Chụp ảnh", label_visibility="collapsed")
        if photo:
            input_image = Image.open(photo).convert("RGB")

    if input_image:
        st.image(input_image, caption="Ảnh đã chọn", use_container_width=True)
    else:
        st.markdown(
            '<div class="placeholder"><span>Tải ảnh lên hoặc chụp ảnh để bắt đầu</span></div>',
            unsafe_allow_html=True,
        )

    classify = st.button("RUN", disabled=(input_image is None))

    if classify and input_image:
        with st.spinner("Đang xử lý…"):
            if method_choice == "ConvNeXtV2-Tiny":
                m, _ = load_conv(CONV_PATH)
                st.session_state.result = predict_conv(m, input_image)
            elif method_choice == "Fusion":
                m, _ = load_fusion(FUS_CONVNEXT, FUS_FASTVIT, FUS_HEAD)
                st.session_state.result = predict_fusion(m, input_image)
            else:
                m, _ = load_ens(ENS_CONVNEXT, ENS_FASTVIT, ENS_MOBILE, ENS_DINO)
                st.session_state.result = predict_ensemble(m, input_image)
            st.session_state.method_used = method_choice
        st.rerun()

# ── Kết quả ───────────────────────────────────────────
else:
    if st.button("Quay lại", key="back"):
        st.session_state.result      = None
        st.session_state.method_used = None
        st.rerun()

result = st.session_state.result
if result:
    conf      = result["top1_confidence"] * 100
    demo_note = (
        '<div class="demo-note">⚠️ Demo mode — kết quả ngẫu nhiên</div>'
        if result["is_dummy"] else ""
    )
    method_tag = f'<div class="method-tag">🔬 {result.get("method","")}</div>'
    st.markdown(
        f"""
        <div class="result-banner">
            <div class="tree-name">{result['top1_class']}</div>
            <div class="confidence">Độ tin cậy mô hình: {conf:.1f}%</div>
            {method_tag}
            {demo_note}
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown("#### Thông tin chi tiết cây xanh:")
    render_tree_info(result["top1_class"], info_db)
