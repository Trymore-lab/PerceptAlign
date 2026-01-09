import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from math import gcd
from torch.utils.checkpoint import checkpoint


def convert_bn_to_gn(module, num_groups=32):
    """Convert all BatchNorm2d layers to GroupNorm for stable small-batch training."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            if num_channels % num_groups == 0:
                gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
                setattr(module, name, gn)
            else:
                g = gcd(num_channels, num_groups)
                gn = nn.GroupNorm(num_groups=g, num_channels=num_channels)
                setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups)


class NeRFPositionalEncoding(nn.Module):
    """
    Multi-frequency positional encoding as in Eq. (22):
    Φ(p) = [sin(2^k π p), cos(2^k π p)]_{k=0}^{K-1}
    """
    def __init__(self, depth=10):
        super().__init__()
        self.depth = depth
        self.output_dim = depth * 2 * 3 + 3  # sin + cos for each freq, plus original
        freq_bands = 2.**torch.linspace(0., depth - 1, steps=depth)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        # x: [..., 3] -> [..., output_dim]
        x_proj = x.unsqueeze(-1) * self.freq_bands * np.pi  # [..., 3, depth]
        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)
        x_encoded = torch.cat(
            [x_sin.reshape(*x.shape[:-1], -1), x_cos.reshape(*x.shape[:-1], -1)],
            dim=-1,
        )
        x_output = torch.cat([x, x_encoded], dim=-1)
        return x_output


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm, compatible with gradient checkpointing."""
    def __init__(self, d_model, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, key_padding_mask=None):
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class posenet(nn.Module):
    """
    WiFi-based 3D pose estimation network following the paper design.
    
    Token Construction (Eq. 24):
        u_{n,t} = LayerNorm(W_f·f_{n,t} + W_e·e_n + r_t + s_n)
    
    where:
        - f_{n,t}: CNN feature for receiver n at time t
        - e_n: spatial embedding from physical 3D coordinates (Eq. 23)
        - r_t: temporal embedding (shared across receivers)
        - s_n: receiver-specific bias
    
    Full Sequence (Eq. 25):
        U^{(0)} = {u_{n,t}}_{n=1,t=1}^{N_r,T}
        
    Sequence length = N_r × T (e.g., 3 × 120 = 360 tokens)
    
    The Transformer performs joint spatio-temporal attention over all tokens,
    enabling both cross-receiver and cross-time interactions.
    """
    def __init__(self, num_keypoints, rel_rx_coords, num_layers, num_heads, pos_enc_depth, max_seq_len=180):
        super(posenet, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_receivers = rel_rx_coords.shape[0]  # N_r = 3
        self.max_seq_len = max_seq_len
        self.d_model = 512
        
        # ========== CNN Backbone (ResNet34 with GroupNorm) ==========
        resnet_raw_model = torchvision.models.resnet34(weights=None)
        convert_bn_to_gn(resnet_raw_model)
        
        self.encoder_conv1 = resnet_raw_model.conv1
        self.encoder_bn1 = resnet_raw_model.bn1
        self.encoder_relu = resnet_raw_model.relu
        self.encoder_maxpool = resnet_raw_model.maxpool
        self.encoder_layer1 = resnet_raw_model.layer1
        self.encoder_layer2 = resnet_raw_model.layer2
        self.encoder_layer3 = resnet_raw_model.layer3
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ========== Feature Projection: W_f in Eq. (24) ==========
        self.feature_proj = nn.Linear(256, self.d_model)
        
        # ========== Spatial Embedding: e_n = g_ψ(Φ(p_n)) in Eq. (23) ==========
        # Φ: multi-frequency encoding (Eq. 22)
        self.pos_encoder = NeRFPositionalEncoding(depth=pos_enc_depth)
        nerf_dim = self.pos_encoder.output_dim
        
        # g_ψ: MLP projection (W_e in Eq. 24)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(nerf_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # Compute and store default spatial embeddings for each receiver (scene-specific).
        # For cross-layout/scenes, you can override by passing rel_rx_coords to forward().
        with torch.no_grad():
            rx_encoded = self.pos_encoder(rel_rx_coords)  # [N_r, nerf_dim]
        self.register_buffer("rx_coords_encoded", rx_encoded)
        # Note: spatial_mlp will be applied in forward to get e_n
        
        # ========== Temporal Embedding: r_t in Eq. (24) ==========
        # Shared across all receivers
        self.temporal_embedding = nn.Parameter(torch.zeros(1, max_seq_len, self.d_model))
        nn.init.normal_(self.temporal_embedding, std=0.02)
        
        # ========== Receiver-specific Bias: s_n in Eq. (24) ==========
        self.receiver_bias = nn.Parameter(torch.zeros(1, 1, self.num_receivers, self.d_model))
        nn.init.normal_(self.receiver_bias, std=0.02)
        
        # ========== Token LayerNorm: as in Eq. (24) ==========
        self.token_norm = nn.LayerNorm(self.d_model)
        
        # ========== Transformer Encoder ==========
        # Operates on N_r × T tokens with joint spatio-temporal attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, num_heads, mlp_ratio=4, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.transformer_norm = nn.LayerNorm(self.d_model)
        
        # ========== Keypoint Decoder ==========
        # From fused tokens to K keypoints per frame
        # We use a simple MLP that processes each frame's aggregated representation
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model * self.num_receivers, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_keypoints * 3)
        )

    def _apply_cnn(self, x):
        """Apply CNN backbone to extract features."""
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        return x

    def forward(self, x, mask=None, rel_rx_coords=None, rx_mask=None):
        B, T, N_r, C, H, W = x.shape  # T = sequence length, N_r = num receivers
        
        # ========== CNN Feature Extraction (chunked for memory) ==========
        cnn_chunk_size = 8
        all_feats_chunks = []
        for i in range(0, T, cnn_chunk_size):
            chunk = x[:, i:i+cnn_chunk_size, :, :, :, :]
            _B, _T, _N_r, _C, _H, _W = chunk.shape
            
            # Reshape: [B, T_chunk, N_r, C, H, W] -> [B*T_chunk*N_r, C, H, W]
            chunk_reshaped = chunk.reshape(_B * _T * _N_r, _C, _H, _W)
            
            # CNN forward
            chunk_feats = self._apply_cnn(chunk_reshaped)  # [B*T*N_r, 256, H', W']
            
            # Global Average Pooling: -> [B*T*N_r, 256]
            chunk_feats = self.global_pool(chunk_feats).flatten(1)
            chunk_feats = chunk_feats.view(_B, _T, _N_r, 256)
            all_feats_chunks.append(chunk_feats)
        
        # Concatenate: [B, T, N_r, 256]
        f = torch.cat(all_feats_chunks, dim=1)
        
        # ========== Token Construction (Eq. 24) ==========
        # W_f · f_{n,t}: Feature projection
        f_proj = self.feature_proj(f)  # [B, T, N_r, D]
        
        # e_n = g_ψ(Φ(p_n)): Spatial embedding from physical coordinates
        if rel_rx_coords is None:
            # Default: use the receiver layout provided at init-time
            e_n = self.spatial_mlp(self.rx_coords_encoded)  # [N_r, D]
            e_n = e_n.view(1, 1, N_r, self.d_model)  # [1, 1, N_r, D] for broadcasting
        else:
            # Dynamic layout: rel_rx_coords can be [N_r, 3] or [B, N_r, 3]
            if rel_rx_coords.dim() == 2:
                rel_rx_coords = rel_rx_coords.unsqueeze(0).expand(B, -1, -1)  # [B, N_r, 3]
            if rel_rx_coords.dim() != 3 or rel_rx_coords.size(0) != B:
                raise ValueError(f"rel_rx_coords must be [N_r,3] or [B,N_r,3], got {tuple(rel_rx_coords.shape)}")
            if rel_rx_coords.size(1) != N_r or rel_rx_coords.size(2) != 3:
                raise ValueError(f"rel_rx_coords shape mismatch: expected [B,{N_r},3], got {tuple(rel_rx_coords.shape)}")

            rx_flat = rel_rx_coords.reshape(B * N_r, 3)
            rx_encoded = self.pos_encoder(rx_flat).view(B, N_r, -1)  # [B, N_r, nerf_dim]
            e_n = self.spatial_mlp(rx_encoded)  # [B, N_r, D]
            e_n = e_n.unsqueeze(1)  # [B, 1, N_r, D] for broadcasting
        
        # r_t: Temporal embedding (shared across receivers)
        r_t = self.temporal_embedding[:, :T, :]  # [1, T, D]
        r_t = r_t.unsqueeze(2)  # [1, T, 1, D] for broadcasting
        
        # s_n: Receiver-specific bias
        s_n = self.receiver_bias  # [1, 1, N_r, D]
        
        # Combine: u_{n,t} = LayerNorm(W_f·f_{n,t} + W_e·e_n + r_t + s_n)
        u = f_proj + e_n + r_t + s_n  # [B, T, N_r, D]
        u = self.token_norm(u)
        
        # ========== Reshape to Sequence (Eq. 25) ==========
        # U^{(0)} = {u_{n,t}}_{n=1,t=1}^{N_r,T}
        # Flatten T and N_r dimensions: [B, T, N_r, D] -> [B, T*N_r, D]
        u = u.view(B, T * N_r, self.d_model)  # [B, T*N_r, D]

        # Padding-aware attention mask: key_padding_mask=True means "ignore"
        if rx_mask is None:
            rx_mask = torch.ones((B, N_r), dtype=torch.bool, device=u.device)
        else:
            if rx_mask.dtype != torch.bool:
                rx_mask = rx_mask.bool()
            if rx_mask.dim() == 1:
                rx_mask = rx_mask.view(1, -1).expand(B, -1)
            if rx_mask.shape != (B, N_r):
                raise ValueError(f"rx_mask must be [B,{N_r}] (or [N_r]), got {tuple(rx_mask.shape)}")

        if mask is None:
            token_mask = rx_mask.unsqueeze(1).expand(B, T, N_r)  # [B,T,N_r]
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            token_mask = mask.unsqueeze(-1) & rx_mask.unsqueeze(1)  # [B,T,N_r]

        key_padding_mask = ~token_mask.reshape(B, T * N_r)
        
        # ========== Transformer: Joint Spatio-Temporal Attention ==========
        for block in self.transformer_blocks:
            # Checkpoint only when gradients are enabled (training); avoids warnings + speeds up eval.
            if self.training and torch.is_grad_enabled():
                u = checkpoint(block, u, key_padding_mask, use_reentrant=True)
            else:
                u = block(u, key_padding_mask=key_padding_mask)
        u = self.transformer_norm(u)  # [B, T*N_r, D]
        
        # ========== Reshape Back for Decoding ==========
        # [B, T*N_r, D] -> [B, T, N_r, D]
        u = u.view(B, T, N_r, self.d_model)
        # Zero-out dropped receivers before aggregation/decoding
        u = u * rx_mask.view(B, 1, N_r, 1).to(dtype=u.dtype)
        
        # ========== Aggregate Receivers per Frame ==========
        # Concatenate N_r tokens for each frame: [B, T, N_r*D]
        u = u.view(B, T, N_r * self.d_model)
        
        # ========== Decode to Keypoints ==========
        output = self.decoder(u)  # [B, T, K*3]
        output = output.view(B, T, self.num_keypoints, 3)
        
        return output


def weights_init(m):
    """Initialize weights for Conv2d, GroupNorm, BatchNorm, and Linear layers."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
