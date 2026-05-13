"""SDPA patch for Swin V2 WindowAttention (baseline comparisons)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from . import swin_transformer_v2 as swin_v2


class WindowAttentionSDPA(swin_v2.WindowAttention):
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape

        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if self.qkv_bias_separate:
                qkv = self.qkv(x)
                qkv += qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        q = q * logit_scale

        rel_pos_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        rel_pos = rel_pos_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        rel_pos = rel_pos.permute(2, 0, 1).contiguous()
        rel_pos = 16 * torch.sigmoid(rel_pos)

        attn_bias = rel_pos.unsqueeze(0)
        if mask is not None:
            num_win = mask.shape[0]
            mask_full = mask.unsqueeze(1).repeat(B_ // num_win, 1, 1, 1)
            attn_bias = attn_bias + mask_full
        attn_bias = attn_bias.to(dtype=q.dtype)
        if attn_bias.shape[0] != B_:
            attn_bias = attn_bias.expand(B_, -1, -1, -1)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = self.attn_drop(attn_out)

        x = attn_out.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


_PATCHED = False


def patch_swin_v2_sdpa() -> None:
    global _PATCHED
    if _PATCHED:
        return
    swin_v2.WindowAttention = WindowAttentionSDPA
    _PATCHED = True
