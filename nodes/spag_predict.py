"""SPAG Predict node for ComfyUI-Sharp."""

import hashlib
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Try to import ComfyUI folder_paths for output directory
try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

from ..utils.image import comfy_to_numpy_rgb
from sharp.utils.spherical import pixel_to_spherical, spherical_to_direction, latitude_scale_factor, get_pixel_grid
from sharp.utils.gaussians import Gaussians3D, save_ply
from sharp.utils import color_space as cs_utils

class SPAGPredict:
    """Run SPAG inference to generate 3D Gaussians from a 360 equirectangular image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SHARP_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "depth_map": ("IMAGE", {
                    "tooltip": "Optional external depth map (must match image dimensions). If provided, SHARP depth prediction is skipped."
                }),
                "depth_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Scale factor for depth values. Increase if scene looks too small/tight."
                }),
                "depth_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Offset added to depth values."
                }),
                "base_scale": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Base scale size for Gaussians. Adjust to fill gaps or reduce overdraw."
                }),
                "use_pole_reconstruction": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Flatten floor and ceiling at poles to reduce artifacts."
                }),
                "floor_threshold": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.5,
                    "max": 0.99,
                    "step": 0.01,
                    "tooltip": "Vertical coordinate threshold (0-1) for floor flattening."
                }),
                "ceiling_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Vertical coordinate threshold (0-1) for ceiling flattening."
                }),
                "output_prefix": ("STRING", {
                    "default": "spag_360",
                    "tooltip": "Prefix for output PLY filename or folder name."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "EXTRINSICS", "INTRINSICS",)
    RETURN_NAMES = ("ply_path", "extrinsics", "intrinsics",)
    FUNCTION = "predict"
    CATEGORY = "SHARP"
    OUTPUT_NODE = True
    DESCRIPTION = "Generate 3D Gaussians from equirectangular 360° panoramas using the SPAG method. Creates a 3D scene viewable from the center."

    @torch.no_grad()
    def predict(
        self,
        model: dict,
        image: torch.Tensor,
        depth_map: torch.Tensor = None,
        depth_scale: float = 1.0,
        depth_offset: float = 0.0,
        base_scale: float = 0.01,
        use_pole_reconstruction: bool = True,
        floor_threshold: float = 0.85,
        ceiling_threshold: float = 0.15,
        output_prefix: str = "spag_360",
    ):
        predictor = model["predictor"]
        device = torch.device(model["device"])

        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        if depth_map is not None:
             if depth_map.dim() == 3:
                depth_map = depth_map.unsqueeze(0)
             # Resize depth to match image if needed
             if depth_map.shape[1:3] != image.shape[1:3]:
                depth_map = F.interpolate(depth_map.permute(0, 3, 1, 2), size=image.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)


        batch_size = image.shape[0]
        print(f"[SPAG] Processing {batch_size} image(s)")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)

        # Output path handling
        if batch_size == 1:
            output_filename = f"{output_prefix}_{timestamp}.ply"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            is_batch = False
        else:
            folder_name = f"{output_prefix}_{timestamp}"
            output_folder = os.path.join(OUTPUT_DIR, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            output_path = output_folder
            is_batch = True

        all_ply_paths = []
        all_extrinsics = []
        all_intrinsics = []
        inference_start = time.time()

        for i in range(batch_size):
            single_image = image[i:i+1] # [1, H, W, 3]
            
            # Use external depth if provided, else predict
            if depth_map is not None:
                print(f"[SPAG] Using provided depth map for image {i+1}...")
                depth_t = depth_map[i:i+1, ..., 0].to(device) # [1, H, W]
                
                # Apply scale and offset
                depth_t = depth_t * depth_scale + depth_offset
                
                # Avoid zero/negative depth which would cause collapse to origin
                depth_t = torch.maximum(depth_t, torch.tensor(0.1, device=device))
                H, W = single_image.shape[1], single_image.shape[2]
                dummy_f_px = max(H, W)
                
            else:
                print(f"[SPAG] Predicting depth for image {i+1}...")
                
                img_np = comfy_to_numpy_rgb(single_image)
                height, width = img_np.shape[:2]
                
                # Prepare image for SHARP
                internal_shape = (1536, 1536)
                image_pt = torch.from_numpy(img_np.copy()).float().to(device).permute(2, 0, 1) / 255.0 # [3, H, W]
                
                image_resized_pt = F.interpolate(
                    image_pt[None],
                    size=(internal_shape[1], internal_shape[0]),
                    mode="bilinear",
                    align_corners=True,
                ) # [1, 3, Hint, Wint]
                
                # Encode
                monodepth_output, _ = predictor.encode(image_resized_pt)
                
                # Decode with dummy focal length
                f_px = max(width, height) 
                disparity_factor = torch.tensor([f_px / width]).float().to(device)
                
                gaussians_ndc = predictor.decode(monodepth_output, image_resized_pt, disparity_factor)
                
                # Recalculate intrinsics for unprojection
                intrinsics = torch.tensor([
                    [f_px, 0, width / 2, 0],
                    [0, f_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]).float().to(device)
                intrinsics_resized = intrinsics.clone()
                intrinsics_resized[0] *= internal_shape[0] / width
                intrinsics_resized[1] *= internal_shape[1] / height
                
                gaussians_metric = local_unproject(gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape)
                
                # Extract depth
                # SHARP output might be different resolution than input internal_shape
                num_gaussians = gaussians_metric.mean_vectors.shape[1]
                W_int = internal_shape[0]
                H_int = num_gaussians // W_int
                
                if num_gaussians != W_int * H_int:
                     print(f"[SPAG] Warning: Number of gaussians {num_gaussians} is not divisible by internal width {W_int}. Choosing sqrt.")
                     S = int(np.sqrt(num_gaussians))
                     W_int = S
                     H_int = S

                depth_t = gaussians_metric.mean_vectors[0, :, 2].reshape(1, H_int, W_int)
                depth_t = F.interpolate(depth_t.unsqueeze(1), size=(height, width), mode='bilinear').squeeze(1) # [1, H, W]
                
                H, W = height, width
                dummy_f_px = f_px
            
            # --- SPAG GENERATION ---
            
            # 1. Pixel coordinates
            u, v = get_pixel_grid(H, W, device) # [H, W]
            
            # 2. Spherical coordinates
            theta, phi = pixel_to_spherical(u, v) # [H, W]
            
            # 3. Direction vectors (Unit sphere)
            directions = spherical_to_direction(theta, phi) # [H, W, 3]
            
            # 4. Positions = Depth * Direction
            depth_per_pixel = depth_t.squeeze(0) # [H, W]
            positions = directions * depth_per_pixel.unsqueeze(-1) # [H, W, 3]
            
            # 5. Pole Reconstruction (Optional)
            if use_pole_reconstruction:
                is_floor = v > floor_threshold
                is_ceiling = v < ceiling_threshold
                
                floor_indices = torch.where(v > floor_threshold)
                if len(floor_indices[0]) > 0:
                     current_y = positions[..., 1]
                     floor_y_target = torch.quantile(current_y[is_floor], 0.5)
                     t_floor = floor_y_target / (directions[..., 1] + 1e-6)
                     positions[is_floor] = directions[is_floor] * t_floor[is_floor].unsqueeze(-1)
                     
                ceiling_indices = torch.where(v < ceiling_threshold)
                if len(ceiling_indices[0]) > 0:
                     ceiling_y_target = torch.quantile(positions[..., 1][is_ceiling], 0.5)
                     t_ceil = ceiling_y_target / (directions[..., 1] + 1e-6)
                     positions[is_ceiling] = directions[is_ceiling] * t_ceil[is_ceiling].unsqueeze(-1)

            # 6. Scaling
            scale_factors = latitude_scale_factor(phi, base_scale=base_scale) # [H, W]
            scales = scale_factors.unsqueeze(-1).expand(-1, -1, 3) # [H, W, 3]

            # 7. Colors
            colors_srgb = single_image.squeeze(0).to(device) # [H, W, 3]
            colors_linear = cs_utils.sRGB2linearRGB(colors_srgb)
            
            # 8. Rotations
            quaternions = torch.zeros(H, W, 4, device=device)
            quaternions[..., 0] = 1.0 
            
            # 9. Opacity
            opacities = torch.ones(H, W, device=device)

            # Flatten
            num_points = H * W
            flat_positions = positions.reshape(1, num_points, 3)
            flat_scales = scales.reshape(1, num_points, 3)
            flat_quats = quaternions.reshape(1, num_points, 4)
            flat_colors = colors_linear.reshape(1, num_points, 3)
            flat_opacities = opacities.reshape(1, num_points)

            # Create Gaussians3D object
            gaussians = Gaussians3D(
                mean_vectors=flat_positions,
                singular_values=flat_scales,
                quaternions=flat_quats,
                colors=flat_colors,
                opacities=flat_opacities
            )
            
            # Save
            if is_batch:
                ply_filename = f"{i+1:03d}.ply"
                ply_path = os.path.join(output_folder, ply_filename)
            else:
                ply_path = output_path
            
            _, metadata = save_ply(gaussians, dummy_f_px, (H, W), Path(ply_path))
            
            all_ply_paths.append(ply_path)
            all_extrinsics.append(metadata["extrinsic"])
            all_intrinsics.append(metadata["intrinsic"])

            print(f"[SPAG] Saved {ply_path} ({num_points:,} gaussians)")

        inference_time = time.time() - inference_start
        print(f"[SPAG] Total time: {inference_time:.2f}s")
        
        # Return values
        if is_batch:
            return (output_path, all_extrinsics[0], all_intrinsics[0],)
        else:
            return (output_path, all_extrinsics[0], all_intrinsics[0],)

# Helper to unproject NDC to Metric for the depth prediction part
def local_unproject(gaussians_ndc, extrinsics, intrinsics, image_shape):
    from sharp.utils.gaussians import get_unprojection_matrix, apply_transform
    unprojection_matrix = get_unprojection_matrix(extrinsics, intrinsics, image_shape)
    gaussians = apply_transform(gaussians_ndc, unprojection_matrix[:3])
    return gaussians


NODE_CLASS_MAPPINGS = {
    "SPAGPredict": SPAGPredict,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SPAGPredict": "SPAG Predict (360)",
}
