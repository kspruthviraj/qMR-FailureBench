"""
Model registry — single entry-point to instantiate any architecture by name.
"""

from __future__ import annotations

import torch.nn as nn

from .resnet1d import ResNet1D
from .vit1d import ViT1D
from .spatiotemporal_transformer import SpatioTemporalTransformer

_REGISTRY = {
    "resnet1d_18": lambda cfg: ResNet1D(
        in_channels=cfg.get("input_channels", 2),
        base_channels=64,
        n_blocks=[2, 2, 2, 2],
        hidden_dim=cfg.get("hidden_dim", 256),
        output_dim=cfg.get("output_dim", 2),
        dropout=cfg.get("dropout", 0.1),
        evidential=cfg.get("evidential", False),
    ),
    "resnet1d_50": lambda cfg: ResNet1D(
        in_channels=cfg.get("input_channels", 2),
        base_channels=64,
        n_blocks=[3, 4, 6, 3],
        hidden_dim=cfg.get("hidden_dim", 256),
        output_dim=cfg.get("output_dim", 2),
        dropout=cfg.get("dropout", 0.1),
        evidential=cfg.get("evidential", False),
    ),
    "vit1d": lambda cfg: ViT1D(
        in_channels=cfg.get("input_channels", 2),
        seq_len=cfg.get("seq_len", 2048),
        patch_size=cfg.get("patch_size", 32),
        hidden_dim=cfg.get("hidden_dim", 256),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_transformer_layers", 6),
        output_dim=cfg.get("output_dim", 2),
        dropout=cfg.get("dropout", 0.1),
        evidential=cfg.get("evidential", False),
    ),
    "spatiotemporal_transformer": lambda cfg: SpatioTemporalTransformer(
        in_channels=cfg.get("input_channels", 2),
        seq_len=cfg.get("seq_len", 1000),
        hidden_dim=cfg.get("hidden_dim", 128),
        n_heads=cfg.get("n_heads", 4),
        n_temporal_layers_1=cfg.get("n_temporal_layers_1", 3),
        n_temporal_layers_2=cfg.get("n_temporal_layers_2", 2),
        output_dim=cfg.get("output_dim", 2),
        dropout=cfg.get("dropout", 0.1),
        evidential=cfg.get("evidential", False),
    ),
}


def build_model(name: str, cfg: dict) -> nn.Module:
    """Instantiate a model by name using the config dict."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](cfg)
