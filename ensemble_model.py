"""
ensemble_model.py
─────────────────────────────────────────────────────────────────
Module nhận diện lá cây đường phố Phường Bến Nghé — Cascaded Ensemble

Kiến trúc: 4 model kết hợp
  ┌─────────────────────────────────────────────────────┐
  │  Ảnh đầu vào                                        │
  │       ↓                                             │
  │  MobileNetV4-Small (Gate)                           │
  │       ↓                                             │
  │  confidence ≥ 0.90 ? ──YES──→ Trả kết quả ngay     │
  │       │ NO                                          │
  │       ↓                                             │
  │  4 model song song + TTA 6 transforms               │
  │       ↓                                             │
  │  Class-wise Weighted Ensemble → Kết quả cuối        │
  └─────────────────────────────────────────────────────┘

Sử dụng:
    from ensemble_model import load_ensemble, predict

    models = load_ensemble(
        convnext_path = 'ConvNeXtV2Tiny_S1_CrossEntropy.pth',
        fastvit_path  = 'fastvit_S1_L1_final.pth',
        mobilev4_path = 'best_model.pth',
        dinov2_path   = 'DINOv2_ViTS14_LPFT_Final_ema.pth',
    )
    result = predict(models, image)
    print(result['predicted_class'], result['confidence'])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'BoDe', 'CamXe', 'DauRai', 'GiaTy', 'GiangHuong',
    'GoMat', 'KenHong', 'LimSet', 'LocVung', 'MacNua',
    'MeTay', 'MeXanh', 'NhacNgua', 'SaoDen', 'Si',
    'SoKhi', 'SuDai', 'Trai', 'Viet',
]
NUM_CLASSES     = len(CLASS_NAMES)
IMG_SIZE        = 224
MEAN            = [0.485, 0.456, 0.406]
STD             = [0.229, 0.224, 0.225]
CONF_THRESHOLD  = 0.90   # ngưỡng Cascaded gate

# Class-wise weights W[model_idx, class_idx]
# Tính từ confusion matrix trên val set — ConvNeXtV2, FastViT, MobileNetV4, DINOv2
# (cập nhật lại sau khi chạy notebook 06 để có giá trị chính xác)
_DEFAULT_CLASS_WEIGHTS = np.array([
    [0.35] * NUM_CLASSES,  # ConvNeXtV2
    [0.30] * NUM_CLASSES,  # FastViT
    [0.20] * NUM_CLASSES,  # MobileNetV4
    [0.15] * NUM_CLASSES,  # DINOv2
], dtype=np.float32)
# Normalize theo cột
_DEFAULT_CLASS_WEIGHTS /= _DEFAULT_CLASS_WEIGHTS.sum(0, keepdims=True)


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────
def _base_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def _tta_transforms(img_size: int = IMG_SIZE) -> list:
    """6 phép biến đổi TTA."""
    base = [transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size)]
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return [
        transforms.Compose(base + norm),
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + norm),
        transforms.Compose(base + [transforms.RandomVerticalFlip(p=1.0)]   + norm),
        transforms.Compose(base + [transforms.RandomRotation((15, 15))]    + norm),
        transforms.Compose(base + [transforms.RandomRotation((-15, -15))]  + norm),
        transforms.Compose([transforms.Resize(int(img_size * 1.1)),
                            transforms.CenterCrop(img_size)] + norm),
    ]


# ─────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────
def _load_state(path: str, device: torch.device) -> dict:
    """Load state dict — hỗ trợ cả EMA shadow dict và state_dict thông thường."""
    state = torch.load(path, map_location=device, weights_only=False)
    # Nếu là checkpoint đầy đủ (có key 'model_state')
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # Strip prefix 'module.' nếu train DataParallel
    state = {k.replace('module.', ''): v for k, v in state.items()}
    return state


def load_ensemble(
    convnext_path : str,
    fastvit_path  : str,
    mobilev4_path : str,
    dinov2_path   : str,
    class_weights : np.ndarray = None,
    device        : torch.device = None,
) -> dict:
    """
    Load 4 model vào bộ nhớ, sẵn sàng cho inference.

    Params
    ------
    convnext_path  : path tới ConvNeXtV2Tiny_S1_CrossEntropy.pth
    fastvit_path   : path tới fastvit_S1_L1_final.pth
    mobilev4_path  : path tới best_model.pth  (MobileNetV4)
    dinov2_path    : path tới DINOv2_ViTS14_LPFT_Final_ema.pth
    class_weights  : ma trận trọng số (4 × 19), mặc định dùng uniform
    device         : torch.device, mặc định tự chọn CUDA/CPU

    Returns
    -------
    dict với keys: convnext, fastvit, mobilev4, dinov2, W, device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── ConvNeXtV2-Tiny ───────────────────────────────────────
    print('Loading ConvNeXtV2-Tiny...', end=' ', flush=True)
    convnext = timm.create_model(
        'convnextv2_tiny.fcmae_ft_in22k_in1k',
        pretrained=False, num_classes=NUM_CLASSES)
    convnext.load_state_dict(_load_state(convnext_path, device), strict=False)
    convnext.eval().to(device)
    print('✓')

    # ── FastViT-S12 ───────────────────────────────────────────
    print('Loading FastViT-S12...', end=' ', flush=True)
    fastvit = timm.create_model(
        'fastvit_s12.apple_in1k',
        pretrained=False, num_classes=NUM_CLASSES)
    fastvit.load_state_dict(_load_state(fastvit_path, device), strict=False)
    fastvit.eval().to(device)
    print('✓')

    # ── MobileNetV4-Small ─────────────────────────────────────
    print('Loading MobileNetV4-Small...', end=' ', flush=True)
    mobilev4 = timm.create_model(
        'mobilenetv4_conv_small.e2400_r224_in1k',
        pretrained=False, num_classes=NUM_CLASSES)
    mobilev4.load_state_dict(_load_state(mobilev4_path, device), strict=False)
    mobilev4.eval().to(device)
    print('✓')

    # ── DINOv2-ViT-S/14 ──────────────────────────────────────
    print('Loading DINOv2-ViT-S/14...', end=' ', flush=True)
    dinov2 = timm.create_model(
        'vit_small_patch14_dinov2.lvd142m',
        pretrained=False, num_classes=NUM_CLASSES,
        img_size=IMG_SIZE, dynamic_img_size=True)
    dinov2.load_state_dict(_load_state(dinov2_path, device), strict=False)
    dinov2.eval().to(device)
    print('✓')

    W = class_weights if class_weights is not None else _DEFAULT_CLASS_WEIGHTS.copy()
    # Normalize lại để chắc chắn tổng cột = 1
    W = W / (W.sum(0, keepdims=True) + 1e-8)

    print('─' * 50)
    print(f'✨ Cascaded Ensemble ready | {NUM_CLASSES} classes | threshold={CONF_THRESHOLD}')
    return {
        'convnext' : convnext,
        'fastvit'  : fastvit,
        'mobilev4' : mobilev4,
        'dinov2'   : dinov2,
        'W'        : W,
        'device'   : device,
    }


# ─────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────
def _infer_one(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Forward pass single image tensor → softmax probs (19,)."""
    with torch.no_grad():
        logits = model(tensor)
        return F.softmax(logits, dim=1)[0].cpu().numpy()


def _infer_tta(model: nn.Module, image: Image.Image,
               device: torch.device) -> np.ndarray:
    """TTA 6 transforms → average probs (19,)."""
    aug_probs = []
    for tf in _tta_transforms():
        tensor = tf(image).unsqueeze(0).to(device)
        aug_probs.append(_infer_one(model, tensor))
    return np.mean(aug_probs, axis=0)


def _class_wise_ensemble(probs_list: list, W: np.ndarray) -> np.ndarray:
    """P_final(c) = Σ W_i(c) × P_i(c)"""
    result = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, p in enumerate(probs_list):
        result += W[i] * p
    return result


# ─────────────────────────────────────────────────────────────
# MAIN PREDICT
# ─────────────────────────────────────────────────────────────
def predict(
    models    : dict,
    image     : Image.Image,
    use_tta   : bool = True,
    top_k     : int  = 5,
) -> dict:
    """
    Nhận diện 1 ảnh lá cây với Cascaded Ensemble.

    Params
    ------
    models   : dict từ load_ensemble()
    image    : PIL Image
    use_tta  : dùng TTA ở giai đoạn 2 (Full ensemble), default True
    top_k    : số lớp top trả về

    Returns
    -------
    {
      'predicted_class' : str,          # tên lớp dự đoán
      'confidence'      : float,        # xác suất cao nhất (0–1)
      'top_k_classes'   : list[str],    # top-k tên lớp
      'top_k_probs'     : list[float],  # top-k xác suất
      'all_probs'       : list[float],  # 19 xác suất đầy đủ
      'path'            : str,          # 'fast' hoặc 'full'
      'is_dummy'        : False,
    }
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    device   = models['device']
    W        = models['W']
    mobilev4 = models['mobilev4']

    # ── Giai đoạn 1: Fast path (MobileNetV4 gate) ────────────
    base_tensor = _base_transform()(image).unsqueeze(0).to(device)
    mob_probs   = _infer_one(mobilev4, base_tensor)
    mob_conf    = mob_probs.max()

    if mob_conf >= CONF_THRESHOLD:
        # Đủ tự tin → trả kết quả ngay
        final_probs = mob_probs
        path_used   = 'fast'
    else:
        # ── Giai đoạn 2: Full ensemble + TTA ─────────────────
        model_list = [
            models['mobilev4'],
            models['dinov2'],
            models['convnext'],
            models['fastvit'],
        ]
        probs_list = []
        for m in model_list:
            if use_tta:
                probs_list.append(_infer_tta(m, image, device))
            else:
                probs_list.append(_infer_one(m, base_tensor))

        final_probs = _class_wise_ensemble(probs_list, W)
        path_used   = 'full'

    # ── Top-k ────────────────────────────────────────────────
    top_idx    = np.argsort(final_probs)[::-1][:top_k]
    top_names  = [CLASS_NAMES[i] for i in top_idx]
    top_probs  = [float(final_probs[i]) for i in top_idx]

    return {
        'predicted_class' : CLASS_NAMES[top_idx[0]],
        'confidence'      : float(final_probs[top_idx[0]]),
        'top_k_classes'   : top_names,
        'top_k_probs'     : top_probs,
        'all_probs'       : final_probs.tolist(),
        'path'            : path_used,
        'is_dummy'        : False,
    }


def predict_batch(
    models  : dict,
    images  : list,
    use_tta : bool = False,
    top_k   : int  = 5,
) -> list:
    """
    Nhận diện nhiều ảnh cùng lúc.

    Params
    ------
    images : list[PIL.Image]
    use_tta: default False cho batch (nhanh hơn)

    Returns
    -------
    list[dict] — mỗi phần tử là kết quả của predict()
    """
    return [predict(models, img, use_tta=use_tta, top_k=top_k)
            for img in images]


def update_class_weights(models: dict, W: np.ndarray) -> None:
    """
    Cập nhật class-wise weights sau khi có kết quả từ notebook 06.

    Params
    ------
    W : np.ndarray shape (4, 19) — precision matrix từ confusion matrix
    """
    W_norm = W / (W.sum(0, keepdims=True) + 1e-8)
    models['W'] = W_norm
    print('Class-wise weights updated.')


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os

    print('=== Quick Test — ensemble_model.py ===')
    print('Usage:')
    print('  from ensemble_model import load_ensemble, predict')
    print('  models = load_ensemble(')
    print('      convnext_path = "ConvNeXtV2Tiny_S1_CrossEntropy.pth",')
    print('      fastvit_path  = "fastvit_S1_L1_final.pth",')
    print('      mobilev4_path = "best_model.pth",')
    print('      dinov2_path   = "DINOv2_ViTS14_LPFT_Final_ema.pth",')
    print('  )')
    print('  result = predict(models, Image.open("leaf.jpg"))')
    print('  print(result)')
    print()
    print(f'NUM_CLASSES    : {NUM_CLASSES}')
    print(f'CLASS_NAMES    : {CLASS_NAMES}')
    print(f'CONF_THRESHOLD : {CONF_THRESHOLD}')
    print(f'IMG_SIZE       : {IMG_SIZE}')
