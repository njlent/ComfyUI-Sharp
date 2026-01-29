"""PanDAPredict node for ComfyUI-Sharp."""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision.transforms import Compose

from ..panda.transforms import Resize, NormalizeImage, PrepareForNet


class PanDAPredict:
    """Run PanDA inference to generate 360 depth map from equirectangular image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PANDA_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "height": ("INT", {
                    "default": 504,
                    "min": 252,
                    "max": 1008,
                    "step": 14,
                    "tooltip": "Processing height (width = 2x height). Higher = better quality but more VRAM."
                }),
                "normalize_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize output depth to 0-1 range."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "predict"
    CATEGORY = "SHARP"
    DESCRIPTION = "Generate 360° depth map from equirectangular panorama using PanDA."

    @torch.no_grad()
    def predict(
        self,
        model: dict,
        image: torch.Tensor,
        height: int = 504,
        normalize_output: bool = True,
    ):
        panda_model = model["model"]
        device = torch.device(model["device"])

        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]
        print(f"[PanDA] Processing {batch_size} image(s) at {height}x{height*2} resolution")

        # Create transform pipeline
        erp_width = height * 2
        transform = Compose([
            Resize(
                width=erp_width,
                height=height,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        depth_maps = []

        for i in range(batch_size):
            single_image = image[i]  # [H, W, 3]
            
            # Convert to numpy RGB [0, 1]
            if single_image.max() > 1.0:
                img_np = single_image.cpu().numpy() / 255.0
            else:
                img_np = single_image.cpu().numpy()
            
            original_h, original_w = img_np.shape[:2]
            
            # Apply transforms
            sample = transform({"image": img_np})
            img_tensor = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
            
            # Run inference
            output = panda_model(img_tensor)
            
            # Debug: print output structure
            if isinstance(output, dict):
                print(f"[PanDA] Output keys: {output.keys()}")
                depth = output["pred_depth"]
            else:
                print(f"[PanDA] Output type: {type(output)}, shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
                depth = output
            
            print(f"[PanDA] Raw depth shape: {depth.shape}, min: {depth.min().item():.4f}, max: {depth.max().item():.4f}")
            
            # Handle different output formats
            if depth.dim() == 4:
                depth = depth[0, 0]  # [H, W]
            elif depth.dim() == 3:
                depth = depth[0]  # [H, W]
            
            # Resize back to original resolution
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(original_h, original_w),
                mode='bilinear',
                align_corners=True
            )[0, 0]
            
            print(f"[PanDA] Resized depth min: {depth.min().item():.4f}, max: {depth.max().item():.4f}")
            
            # Normalize to 0-1 if requested
            if normalize_output:
                depth_min = depth.min()
                depth_max = depth.max()
                if depth_max > depth_min:
                    depth = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth = torch.zeros_like(depth)
            
            # Convert to 3-channel grayscale for ComfyUI IMAGE format
            depth_3ch = depth.unsqueeze(-1).expand(-1, -1, 3)
            depth_maps.append(depth_3ch)

        # Stack batch
        result = torch.stack(depth_maps, dim=0).cpu()
        
        print(f"[PanDA] Depth estimation complete. Output shape: {result.shape}, range: [{result.min():.4f}, {result.max():.4f}]")
        
        return (result,)


NODE_CLASS_MAPPINGS = {
    "PanDAPredict": PanDAPredict,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PanDAPredict": "PanDA Predict (360 Depth)",
}
