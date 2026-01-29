
import torch
import numpy as np
from typing import Tuple

def pixel_to_spherical(u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert normalized pixel coordinates (u,v) in [0,1] to spherical coordinates (theta, phi).
    
    Args:
        u: Normalized horizontal coordinate [0,1]
        v: Normalized vertical coordinate [0,1]
        
    Returns:
        theta: Azimuth angle in radians [0, 2pi]
        phi: Elevation angle in radians [0, pi]
    """
    theta = (1.0 - u) * 2.0 * np.pi
    phi = v * np.pi
    return theta, phi

def spherical_to_direction(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical coordinates to unit direction vectors using the convention:
    r = [sin(phi)cos(theta), cos(phi), -sin(phi)sin(theta)]
    
    Note: Y is up (cos(phi)), X and Z are horizontal plane.
    
    Args:
        theta: Azimuth angle
        phi: Elevation angle
        
    Returns:
        direction: Unit vector tensor of shape [..., 3]
    """
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    x = sin_phi * cos_theta
    y = cos_phi
    z = -sin_phi * sin_theta
    
    return torch.stack([x, y, z], dim=-1)

def latitude_scale_factor(phi: torch.Tensor, base_scale: float = 0.01, epsilon: float = 1e-4) -> torch.Tensor:
    """
    Calculate latitude-aware scaling factor to compensate for equirectangular distortion.
    Scale decreases near poles (phi -> 0 or phi -> pi) to maintain uniform visual density.
    
    Args:
        phi: Elevation angle
        base_scale: Base scale factor at equator
        epsilon: Minimum scale factor to avoid zero/singularity
        
    Returns:
        scale: Scale factor tensor
    """
    sin_phi = torch.sin(phi)
    # Ensure sine is positive (phi in [0, pi]) and clamp to epsilon
    scale = base_scale * torch.maximum(torch.abs(sin_phi), torch.tensor(epsilon, device=phi.device))
    return scale

def get_pixel_grid(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a grid of normalized pixel coordinates.
    
    Args:
        height: Image height
        width: Image width
        device: Torch device
        
    Returns:
        u: Horizontal coordinates [Height, Width]
        v: Vertical coordinates [Height, Width]
    """
    y_coords = torch.linspace(0, 1, height, device=device)
    x_coords = torch.linspace(0, 1, width, device=device)
    
    v, u = torch.meshgrid(y_coords, x_coords, indexing='ij')
    return u, v
