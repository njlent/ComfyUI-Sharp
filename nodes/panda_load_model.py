"""LoadPanDAModel node for ComfyUI-Sharp."""

import os
from pathlib import Path

import torch
import torch.nn as nn

# Try to import huggingface_hub for model downloading
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Try to get ComfyUI models directory
try:
    import folder_paths
    MODELS_DIR = os.path.join(folder_paths.models_dir, "panda")
except ImportError:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "panda")


# Model configurations
MODEL_CONFIGS = {
    "small": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "hf_filename": "panda_small.pth",
    },
    "base": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "hf_filename": "panda_base.pth",
    },
    "large": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "hf_filename": "panda_large.pth",
    },
}

HF_REPO_ID = "ZidongC/PanDA"


class LoadPanDAModel:
    """Load PanDA model for 360 depth estimation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_size": (["small", "base", "large"], {
                    "default": "base",
                    "tooltip": "Model size. Larger = better quality but slower and more VRAM."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run inference on."
                }),
            }
        }

    RETURN_TYPES = ("PANDA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SHARP"
    DESCRIPTION = "Load PanDA model for 360° panoramic depth estimation."

    def load_model(self, model_size: str, device: str):
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = MODEL_CONFIGS[model_size]
        
        # Ensure models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Download model if not cached
        model_path = os.path.join(MODELS_DIR, config["hf_filename"])
        if not os.path.exists(model_path):
            print(f"[PanDA] Downloading {model_size} model from HuggingFace...")
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=config["hf_filename"],
                local_dir=MODELS_DIR,
            )
            print(f"[PanDA] Downloaded to {model_path}")
        else:
            print(f"[PanDA] Using cached model: {model_path}")
        
        # Import DepthAnythingV2 - try multiple locations
        DepthAnythingV2 = None
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            pass
        
        if DepthAnythingV2 is None:
            try:
                from depth_anything_v2_metric.depth_anything_v2.dpt import DepthAnythingV2
            except ImportError:
                pass
        
        if DepthAnythingV2 is None:
            raise ImportError(
                "DepthAnythingV2 not found. Please install depth-anything-v2: "
                "pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
            )
        
        # Create model - try with and without max_depth parameter
        print(f"[PanDA] Loading {model_size} model...")
        try:
            # Metric depth variant has max_depth
            model = DepthAnythingV2(
                encoder=config["encoder"],
                features=config["features"],
                out_channels=config["out_channels"],
                max_depth=1.0,
            )
        except TypeError:
            # Standard DAv2 doesn't have max_depth
            model = DepthAnythingV2(
                encoder=config["encoder"],
                features=config["features"],
                out_channels=config["out_channels"],
            )
        
        # Load weights
        model_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle DataParallel wrapper if present
        if any(key.startswith('module.') for key in model_dict.keys()):
            model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
        
        # Handle 'core.' prefix from PanDA wrapper
        if any(key.startswith('core.') for key in model_dict.keys()):
            model_dict = {k.replace('core.', ''): v for k, v in model_dict.items()}
        
        model.load_state_dict(model_dict, strict=False)
        model.to(device)
        model.eval()
        
        print(f"[PanDA] Model loaded successfully on {device}")
        
        return ({
            "model": model,
            "device": device,
            "config": config,
            "model_size": model_size,
        },)


NODE_CLASS_MAPPINGS = {
    "LoadPanDAModel": LoadPanDAModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPanDAModel": "Load PanDA Model (360 Depth)",
}
