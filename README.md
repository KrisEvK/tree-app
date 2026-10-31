# Urban Tree Species Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Streamlit app that identifies 19 street tree species in Bến Nghé Ward (District 1, Ho Chi Minh City) from a leaf photo. Upload an image or use the camera; the app returns the species name, confidence, conservation status, and botanical notes.

This is the capstone project (Đồ án chuyên ngành) at HUIT's Faculty of Information Technology, extending our earlier conference paper to 19 species with modern architectures and a two-stage ensemble.

---

## Citation

> **Ứng dụng kỹ thuật Deep Learning trong phân loại cây xanh phường Sài Gòn**
> Pham Nguyen Minh Hau, **Nguyen Huu Tri**, Nguyen Viet Truong Toan; Advisors: TS. Ngo Duong Ha, ThS. Tran Nhu Y.
> Faculty of Information Technology Scientific Conference, HUIT, May 2026.

The original paper covered 7 species with a baseline CNN. This release expands to 19 species, replaces the backbone, and adds an ensemble with an early-exit gate.

---

## The data leakage problem

Field data was collected by photographing the same trees from multiple angles on the same day. A random train/val/test split split those near-duplicate images across sets, inflating validation accuracy by ~8 points.

Fix: we grouped all images by individual tree ID before splitting. Every image from a given tree goes entirely into one set. The test numbers below reflect this corrected split.

---

## Architecture

Two inference modes are available in the app:

**ConvNeXtV2-Tiny (single model)**: faster, lower memory.

**Cascaded Ensemble**: runs a fast gate first:

```
Input image (224x224)
        |
MobileNetV4-Small (gate)
        |
confidence >= 0.90? --YES--> return result  (~20 ms on CPU)
        |
       NO
        |
4 models in parallel + TTA (6 transforms)
  ConvNeXtV2-Tiny
  FastViT-S12
  MobileNetV4
  DINOv2-ViTS14 (LP-FT)
        |
class-wise weighted voting
        |
Final prediction
```

Linear Probing then Fine-Tuning (LP-FT) on DINOv2: we froze the backbone and trained only the head first, then unfroze and fine-tuned end-to-end. This prevented the randomly initialized head from corrupting the pretrained features early in training, lifting accuracy from 64.8% (LP only) to 88.1%.

---

## Results

Evaluated on the leak-free grouped test split, 19 species:

| Model | Accuracy | Macro F1 | Latency (CPU) |
| :--- | :---: | :---: | :---: |
| ResNet-50 (baseline) | 89.20% | 0.8845 | ~45 ms |
| MobileNetV4-Small | 91.40% | 0.9082 | ~18 ms |
| FastViT-S12 | 93.65% | 0.9310 | ~28 ms |
| ConvNeXtV2-Tiny | 94.80% | 0.9450 | ~52 ms |
| DINOv2-ViTS14 (LP only) | 64.80% | 0.6350 | ~70 ms |
| DINOv2-ViTS14 (LP-FT) | 88.10% | 0.8790 | ~70 ms |
| Weighted Fusion (ConvNeXt + FastViT) | 96.15% | 0.9580 | ~75 ms |
| Cascaded Ensemble (ours) | **98.35%** | **0.9831** | ~20 ms avg |

---

## Species covered

19 street trees endemic to Bến Nghé Ward. Full botanical data (scientific name, conservation status, morphological traits, distribution) is in `info19.json`.

| # | Vietnamese name | Scientific name |
| :---: | :--- | :--- |
| 1 | Bồ Đề | *Ficus religiosa* |
| 2 | Căm Xe | *Xylia xylocarpa* |
| 3 | Dầu Rái | *Dipterocarpus alatus* |
| 4 | Giá Tỵ | *Tectona grandis* |
| 5 | Giáng Hương | *Pterocarpus macrocarpus* |
| 6 | Gõ Mật | *Sindora siamensis* |
| 7 | Kèn Hồng | *Tabebuia rosea* |
| 8 | Lim Xẹt | *Peltophorum pterocarpum* |
| 9 | Lộc Vừng | *Barringtonia acutangula* |
| 10 | Mác Nưa | *Diospyros mollis* |
| 11 | Me Tây | *Samanea saman* |
| 12 | Me Xanh | *Tamarindus indica* |
| 13 | Nhạc Ngựa | *Swietenia mahagoni* |
| 14 | Sao Đen | *Hopea odorata* |
| 15 | Cây Si | *Ficus benjamina* |
| 16 | Sọ Khỉ | *Khaya senegalensis* |
| 17 | Sứ Đại | *Plumeria rubra* |
| 18 | Cây Trai | *Garcinia fragraeoides* |
| 19 | Viết | *Mimusops elengi* |

---

## File layout

```
tree_app/
├── app.py               # Streamlit UI and inference routing
├── ensemble_model.py    # Cascaded ensemble: gate, TTA, weighted voting
├── fusion_model.py      # Weighted feature fusion head (ConvNeXt + FastViT)
├── info19.json          # Botanical metadata for 19 species
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

Model weights are tracked with Git LFS (`.pth` files). Clone normally; Git LFS pulls them automatically if installed.

---

## Run with Docker

```bash
docker build -t tree-app .
docker run --rm -p 7860:7860 tree-app
```

Open `http://localhost:7860`.

## Run locally

```bash
git clone https://github.com/KrisEvK/tree-app.git
cd tree-app
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Team

- **Nguyen Huu Tri** ([@KrisEvK](https://github.com/KrisEvK)): ML pipeline and ensemble engineering
- Pham Nguyen Minh Hau, Nguyen Viet Truong Toan: data collection, training experiments
- Advisors: TS. Ngo Duong Ha, ThS. Tran Nhu Y (Faculty of Information Technology, HUIT)
