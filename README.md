---
title: Phân Loại Cây Xanh Phường Bến Nghé
emoji: 🌿
colorFrom: green
colorTo: teal
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---

# 🌿 Phân Loại Cây Xanh Phường Bến Nghé

Ứng dụng nhận diện **19 loài cây xanh đường phố** tại Phường Bến Nghé, TP.HCM  
sử dụng kỹ thuật Deep Learning.

## Hai phương pháp phân loại

| | Method 1 | Method 2 |
|---|---|---|
| **Tên** | ConvNeXtV2-Tiny | Fusion Feature |
| **Kiến trúc** | Single model | ConvNeXtV2-Tiny + FastViT-S12 → WeightedFusionClassifier |
| **Đầu vào** | 1 file `.pth` | 3 file `.pth` |
| **Ghi chú** | Nhanh, nhẹ | Độ chính xác cao hơn |

> Chỉ được bật **một** phương pháp tại một thời điểm.

---

## Cấu trúc thư mục

```
tree_app/
├── app.py                              # Streamlit app chính
├── fusion_model.py                     # Module Fusion (Method 2)
├── info19.json                         # Dữ liệu 19 loài cây
├── requirements.txt
├── README.md
│
│  ── Model weights (cần upload thủ công) ──
├── ConvNeXtV2Tiny_S1_CrossEntropy.pth  # Method 1
├── M2_ConvNeXtV2Tiny_S1_CrossEntropy.pth  # Method 2 – backbone ConvNeXt
├── M2_FastViT_S1_CrossEntropy.pth         # Method 2 – backbone FastViT
└── fusion_best.pth                        # Method 2 – Fusion head
```

---

## Hướng dẫn deploy

### A. HuggingFace Spaces

1. Tạo Space mới → chọn **Streamlit** SDK
2. Clone repo Space về máy:
   ```bash
   git clone https://huggingface.co/spaces/<username>/<space-name>
   ```
3. Copy toàn bộ nội dung thư mục `tree_app/` vào repo
4. Upload các file `.pth` (dùng HF web UI hoặc `git lfs`):
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git add *.pth app.py fusion_model.py info19.json requirements.txt README.md
   git commit -m "Initial deploy"
   git push
   ```

> **Lưu ý:** File `.pth` thường > 100MB → **bắt buộc dùng Git LFS** trên HuggingFace.

### B. Streamlit Cloud (github)

1. Push toàn bộ `tree_app/` lên một **GitHub repo public**
2. Vào [share.streamlit.io](https://share.streamlit.io) → New app
3. Chọn repo, branch `main`, **Main file path** = `app.py`
4. Deploy

> **Lưu ý:** GitHub giới hạn file < 100MB. Với file `.pth` lớn hơn, dùng **Git LFS** hoặc host model trên HuggingFace Hub rồi download trong `app.py` (xem phần bên dưới).

### C. Tự động download model từ HuggingFace Hub (khuyến nghị cho Streamlit Cloud)

Nếu file `.pth` quá lớn cho GitHub, thêm đoạn sau vào đầu `app.py`:

```python
from huggingface_hub import hf_hub_download
import os

def ensure_model(repo_id: str, filename: str):
    if not os.path.exists(filename):
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=".")

ensure_model("your-username/your-model-repo", "ConvNeXtV2Tiny_S1_CrossEntropy.pth")
ensure_model("your-username/your-model-repo", "M2_ConvNeXtV2Tiny_S1_CrossEntropy.pth")
ensure_model("your-username/your-model-repo", "M2_FastViT_S1_CrossEntropy.pth")
ensure_model("your-username/your-model-repo", "fusion_best.pth")
```

Thêm vào `requirements.txt`:
```
huggingface_hub>=0.22.0
```

---

## 19 loài cây được nhận diện

Bồ Đề · Cẩm Xe · Dầu Rái · Giá Tị · Giáng Hương · Gõ Mật · Kèn Hồng ·
Lim Sẹt · Lộc Vừng · Mắc Nưa · Mè Tây · Me Xanh · Nhạc Ngựa · Sao Đen ·
Si · Sồi Khỉ · Sứ Đại · Trái · Việt
