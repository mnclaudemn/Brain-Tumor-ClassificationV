import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM implementation for CNN-based models (e.g., ResNet50).

    Highlights image regions that contribute most to prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # hooks
        self._register_hooks()

    # ----------------------------
    # HOOKS
    # ----------------------------
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    # ----------------------------
    # GENERATE HEATMAP
    # ----------------------------
    def generate(self, image, class_idx=None):
        """
        image: (1, 3, H, W) tensor
        """

        self.model.eval()

        output = self.model(image)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # backprop target class
        self.model.zero_grad()
        output[0, class_idx].backward()

        # gradients & activations
        gradients = self.gradients          # (B, C, H, W)
        activations = self.activations      # (B, C, H, W)

        # global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3))  # (B, C)

        # weighted sum of feature maps
        cam = torch.zeros(activations.shape[2:], device=image.device)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]

        cam = F.relu(cam)

        # normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

    # ----------------------------
    # OVERLAY HEATMAP ON IMAGE
    # ----------------------------
    def overlay(self, image_np, cam, alpha=0.5):
        """
        image_np: (H, W, 3) numpy image
        cam: (H, W) heatmap
        """

        heatmap = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlayed = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

        return overlayed
