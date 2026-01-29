"""PanDA model implementation with LoRA support for 360 depth estimation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace


class _LoRA_qkv(nn.Module):
    """LoRA adapter for QKV attention layers."""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


def apply_lora_to_model(da_model, r: int = 4, lora_layer=None):
    """Apply LoRA to a DepthAnythingV2 model in-place."""
    
    if lora_layer is None:
        lora_layer = list(range(len(da_model.pretrained.blocks)))
    
    w_As = []
    w_Bs = []

    # Freeze base model
    for param in da_model.pretrained.parameters():
        param.requires_grad = False

    # Apply LoRA to each transformer block
    for t_layer_i, blk in enumerate(da_model.pretrained.blocks):
        if t_layer_i not in lora_layer:
            continue
        
        w_qkv_linear = blk.attn.qkv
        dim = w_qkv_linear.in_features
        
        w_a_linear_q = nn.Linear(dim, r, bias=False)
        w_b_linear_q = nn.Linear(r, dim, bias=False)
        w_a_linear_v = nn.Linear(dim, r, bias=False)
        w_b_linear_v = nn.Linear(r, dim, bias=False)
        
        w_As.extend([w_a_linear_q, w_a_linear_v])
        w_Bs.extend([w_b_linear_q, w_b_linear_v])
        
        blk.attn.qkv = _LoRA_qkv(
            w_qkv_linear,
            w_a_linear_q,
            w_b_linear_q,
            w_a_linear_v,
            w_b_linear_v,
        )
    
    # Initialize LoRA weights
    for w_A in w_As:
        nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
    for w_B in w_Bs:
        nn.init.zeros_(w_B.weight)
    
    return w_As, w_Bs


class PanDA(nn.Module):
    """PanDA model for panoramic depth estimation."""
    
    def __init__(self, da_model, max_depth=1.0, apply_lora=True, lora_rank=4):
        super().__init__()
        
        self.max_depth = max_depth
        self.core = da_model
        
        if apply_lora:
            self.w_As, self.w_Bs = apply_lora_to_model(da_model, r=lora_rank)
        else:
            self.w_As = []
            self.w_Bs = []

    def forward(self, image, max_depth=None):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Forward through core DAv2
        # Handle different DAv2 API versions
        try:
            erp_pred = self.core(image, self.max_depth)
        except TypeError:
            erp_pred = self.core(image)
        
        # Ensure it's 4D [B, 1, H, W]
        if erp_pred.dim() == 3:
            erp_pred = erp_pred.unsqueeze(1)
        elif erp_pred.dim() == 2:
            erp_pred = erp_pred.unsqueeze(0).unsqueeze(0)

        outputs = {}
        outputs["pred_depth"] = erp_pred * self.max_depth

        return outputs


def create_panda_model(DepthAnythingV2, encoder="vitb", features=128, out_channels=[96, 192, 384, 768], 
                       max_depth=1.0, lora=True, lora_rank=4):
    """Create a PanDA model with LoRA modifications."""
    
    # Create base DAv2 model - try with max_depth first
    try:
        da_model = DepthAnythingV2(
            encoder=encoder,
            features=features,
            out_channels=out_channels,
            max_depth=max_depth,
        )
    except TypeError:
        da_model = DepthAnythingV2(
            encoder=encoder,
            features=features,
            out_channels=out_channels,
        )
    
    # Wrap with PanDA
    panda = PanDA(da_model, max_depth=max_depth, apply_lora=lora, lora_rank=lora_rank)
    
    return panda
