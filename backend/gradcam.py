import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute(self, input_tensor, class_idx):
        self.model.zero_grad()
        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam


def overlay_heatmap(pil_image, cam, size=224):
    """Blend Grad-CAM heatmap onto the scan and return a base64 PNG data URL."""
    image = pil_image.convert("RGB").resize((size, size))
    img_arr = np.array(image).astype(np.float32)

    cam_img = np.uint8(255 * cam)
    cam_resized = np.array(Image.fromarray(cam_img).resize((size, size), Image.BILINEAR)).astype(
        np.float32
    ) / 255.0

    heat = np.zeros((size, size, 3), dtype=np.float32)
    heat[..., 0] = cam_resized
    heat[..., 1] = 0.35 * cam_resized
    heat[..., 2] = 1.0 - cam_resized
    heat *= 255.0

    blended = np.clip(0.55 * img_arr + 0.45 * heat, 0, 255).astype(np.uint8)
    out = Image.fromarray(blended)

    buffer = BytesIO()
    out.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
