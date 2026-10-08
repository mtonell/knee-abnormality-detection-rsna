"""ConvNeXt-Nano 6-View Multi-Head Transformer Architecture & Universal Checkpoint Loader."""

from pathlib import Path
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.constants import CONVNEXT_SERIES_SLOTS, NUM_CLASSES, DEFAULT_IMAGE_SIZE


class ConvNeXt6View(nn.Module):
    """6-View Multi-Head Transformer with ConvNeXt-Nano 2.5D Slab Backbone.

    Architecture Highlights:
    1. Shared Backbone: Evaluates 2.5D tri-slice slabs [s_{t-1}, s_t, s_{t+1}].
    2. Joint Slot + Physical Positional Embeddings: Injects anatomical plane & depth.
    3. Multi-Head Transformer: Cross-slot global attention across all 120 slices.
    4. Pathfinding Query MIL Pooling: Computes pathology-specific attention weights.
    """
    def __init__(
        self,
        backbone_name: str = "convnext_nano.in12k_ft_in1k",
        dim: int = 384,
        depth: int = 2,
        drop: float = 0.2,
        num_classes: int = NUM_CLASSES,
        num_slots: int = len(CONVNEXT_SERIES_SLOTS),
        use_norm: bool = False,
        use_imagenet_norm: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.dim = dim
        self.use_norm = use_norm
        self.use_imagenet_norm = use_imagenet_norm

        try:
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        except Exception:
            # Fallback across timm versions
            fallback_names = ["convnext_nano.in12k_ft_in1k", "convnext_nano", "convnext_tiny"]
            for name in fallback_names:
                try:
                    self.backbone = timm.create_model(name, pretrained=False, num_classes=0)
                    break
                except Exception:
                    continue
        in_features = self.backbone.num_features
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(in_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(num_slots, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()

        self.att = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes)
        )
        self.cls_w = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(num_classes))

    def head(
        self,
        feat: torch.Tensor,
        mask: torch.Tensor,
        pos: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """Applies Transformer self-attention across all slices and computes finding logits."""
        B, S, K, _ = feat.shape
        slot_ids = torch.arange(S, device=feat.device).view(1, S, 1).expand(B, S, K)
        h = self.proj(feat) + self.slot_emb(slot_ids) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        h = h.view(B, S * K, -1)

        pad = ~mask.unsqueeze(-1).expand(B, S, K).reshape(B, S * K)
        pad = pad & ~pad.all(dim=1, keepdim=True)
        h = self.tf(h, src_key_padding_mask=pad)
        h = self.norm(h)

        attn_scores = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, S * K, NUM_CLASSES)
        pooled = torch.einsum("btc,btd->bcd", attn_weights.to(h.dtype), h)
        logits = (pooled * self.cls_w).sum(dim=-1) + self.cls_b

        if return_attention:
            return logits, attn_weights
        return logits

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """Chunked forward pass over 6-view 2.5D tri-slice slabs."""
        if self.use_imagenet_norm:
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3, 1, 1)
            x = (x - mean) / std

        B, S, K, C, H, W = x.shape
        keep = mask.view(-1)
        if not keep.any():
            zero_out = torch.zeros((B, self.num_classes), device=x.device)
            if return_attention:
                return zero_out, torch.zeros((B, S * K, self.num_classes), device=x.device)
            return zero_out

        xv = x.view(B * S, K, C, H, W)[keep].flatten(0, 1).contiguous(memory_format=torch.channels_last)
        chunk_size = 60
        feats = []
        for i in range(0, xv.shape[0], chunk_size):
            feats.append(self.backbone(xv[i:i + chunk_size]))
        f = torch.cat(feats, dim=0)

        feat = f.new_zeros(B * S, K, f.shape[-1])
        feat[keep] = f.view(-1, K, f.shape[-1])
        feat = feat.view(B, S, K, -1)
        return self.head(feat, mask, pos, return_attention=return_attention)


def load_convnext_checkpoint(ckpt_path: Path, device: torch.device) -> ConvNeXt6View:
    """Universal checkpoint loader for ConvNeXt-Nano checkpoints (flagship and fold weights)."""
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = st["model"] if isinstance(st, dict) and "model" in st else st
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    has_norm = any(k.startswith("norm.") for k in state.keys())
    mapped_state = {}
    for k, v in state.items():
        new_k = k
        if new_k.startswith("transformer."):
            new_k = "tf." + new_k[len("transformer."):]
        mapped_state[new_k] = v

    model = ConvNeXt6View(
        backbone_name="convnext_nano.in12k_ft_in1k",
        num_classes=NUM_CLASSES,
        use_norm=has_norm,
        use_imagenet_norm=has_norm
    )
    model.load_state_dict(mapped_state, strict=True)
    model.to(device).eval().to(memory_format=torch.channels_last)
    return model
