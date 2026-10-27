"""
Loads a pretrained deepfake-detection model (no training needed) and
exposes a single predict_image() function used by app.py.

Model: prithivMLmods/Deep-Fake-Detector-v2-Model
- Vision Transformer (ViT) fine-tuned on real/deepfake face images.
- Outputs: "Realism" or "Deepfake" + a confidence score.

For explainability (the premium heatmap feature), we use the model's own
attention weights ("attention rollout") instead of Grad-CAM. Grad-CAM is
built for CNNs; ViT models don't have the same conv-layer structure, but
they DO output attention maps directly, which serve the same purpose —
showing which parts of the image the model focused on.
"""

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None


def load_model():
    """Loads the model once. Call this at server startup, not per-request —
    loading from disk/downloading takes several seconds."""
    global _model, _processor
    if _model is None:
        print(f"Loading {MODEL_NAME} on {_device} ...")
        _processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        _model = ViTForImageClassification.from_pretrained(
            MODEL_NAME, output_attentions=True
        )
        _model.to(_device)
        _model.eval()
        print("Model loaded.")
    return _model, _processor


def predict_image(pil_image: Image.Image, with_heatmap: bool = False):
    """
    Runs one image through the model.

    Returns a dict:
      {
        "label": "Deepfake" | "Realism",
        "confidence": float (0-1),
        "raw_scores": {label: prob, ...},
        "heatmap": np.ndarray (H, W) in [0,1] or None
      }
    """
    model, processor = load_model()
    pil_image = pil_image.convert("RGB")

    inputs = processor(images=pil_image, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]
    predicted_idx = int(torch.argmax(probs).item())
    label = model.config.id2label[predicted_idx]
    confidence = float(probs[predicted_idx].item())

    raw_scores = {
        model.config.id2label[i]: float(probs[i].item())
        for i in range(len(probs))
    }

    heatmap = None
    if with_heatmap:
        heatmap = _attention_rollout(outputs.attentions, image_size=pil_image.size)

    return {
        "label": label,
        "confidence": confidence,
        "raw_scores": raw_scores,
        "heatmap": heatmap,
    }


def _attention_rollout(attentions, image_size):
    """
    Combines attention weights across all transformer layers into a single
    heatmap showing where the model focused, resized to the original image.

    attentions: tuple of tensors, one per layer, each
                shape (batch, heads, tokens, tokens)
    """
    # Average over attention heads for each layer -> (layers, tokens, tokens)
    layer_attns = [a[0].mean(dim=0) for a in attentions]

    # Multiply layers together ("rollout") to propagate attention through depth
    result = torch.eye(layer_attns[0].size(-1))
    for attn in layer_attns:
        attn = attn + torch.eye(attn.size(-1))  # add residual connection
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn, result)

    # Attention from the [CLS] token to every patch token (skip index 0 = CLS itself)
    cls_attention = result[0, 1:]

    num_patches = cls_attention.size(0)
    grid_size = int(num_patches**0.5)
    heatmap = cls_attention.reshape(grid_size, grid_size).numpy()

    # Normalize to 0-1 and resize to the original image dimensions
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(image_size, resample=Image.BILINEAR)

    return np.array(heatmap_img) / 255.0
