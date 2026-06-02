import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
class WeightedFusionClassifier(nn.Module):
    def __init__(self, d1=768, d2=1024, proj_dim=512, num_classes=19, dropout=0.3):
        super().__init__()
        self.proj1 = nn.Linear(d1, proj_dim)        # Project ConvNeXt features
        self.proj2 = nn.Linear(d2, proj_dim)        # Project FastViT features
        self.log_w = nn.Parameter(torch.zeros(2))   
        self.head  = nn.Sequential(
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes)
        )

    def forward(self, f1, f2):
        w     = torch.softmax(self.log_w, dim=0)
        fused = w[0] * self.proj1(f1) + w[1] * self.proj2(f2)
        return self.head(fused)

    def get_weights(self):
        w = torch.softmax(self.log_w.detach(), dim=0)
        return w[0].item(), w[1].item()
def load_fusion_models(
    convnext_path: str,
    fastvit_path: str,
    fusion_path: str,
    device=None
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on device: {device}")
    try:
        print("Loading ConvNeXtV2-Tiny...", end=" ", flush=True)
        convnext = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=False,
            num_classes=0 
        )
        convnext.load_state_dict(
            torch.load(convnext_path, map_location=device),
            strict=False
        )
        convnext.eval().to(device)
        print("Loading FastViT-S12...", end=" ", flush=True)
        fastvit = timm.create_model(
            "fastvit_s12.apple_in1k",
            pretrained=False,
            num_classes=0  # Extract features only
        )
        fastvit.load_state_dict(
            torch.load(fastvit_path, map_location=device),
            strict=False
        )
        fastvit.eval().to(device)
        print("Loading Fusion classifier...", end=" ", flush=True)
        fusion = WeightedFusionClassifier().to(device)
        fusion.load_state_dict(
            torch.load(fusion_path, map_location=device),
            strict=True
        )
        fusion.eval()
        alpha, beta = fusion.get_weights()
        print("─" * 50)
        print(f"✨ Fusion weights: α(ConvNeXt)={alpha:.4f}  β(FastViT)={beta:.4f}")

        return {
            'convnext': convnext,
            'fastvit': fastvit,
            'fusion': fusion,
            'device': device,
            'weights': (alpha, beta),
        }

    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        raise
    except Exception as e:
        print(f"\nError loading models: {e}")
        raise
def create_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
def predict_fusion(
    models: dict,
    image: Image.Image,
    class_names: list = None,
    input_size: int = 224,
    top_k: int = 5
) -> dict:
    convnext = models['convnext']
    fastvit = models['fastvit']
    fusion = models['fusion']
    device = models['device']
    transform = create_transform(input_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        f1 = convnext(tensor)
        f2 = fastvit(tensor)
        logits = fusion(f1, f2)
        probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, min(top_k, len(probs)))

    if class_names is None:
        class_names = [str(i) for i in range(len(probs))]

    result = {
        'predicted_class': class_names[top_idx[0].item()],
        'confidence': top_probs[0].item(),
        'class_names': [class_names[i.item()] for i in top_idx],
        'confidences': [p.item() for p in top_probs],
        'is_dummy': False
    }
    return result