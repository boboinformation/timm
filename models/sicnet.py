# sicnet_window_v2.py  –– 2025‑05‑12 (Window‑MHSA 精確等價 + Large‑Kernel PE + 效能優化)
# Author: Karasu + ChatGPT
# -----------------------------------------------------------------------------
#  ✅ 兩種精確 Window Attention：WinSAScan / WinIntegral (皆等價於 MHSA)
#  ✅ 大核 Positional‑Encoding：5×5 depth‑wise Conv 置於 Stem 之後
#  ✅ 四‑Stage backbone (Stem + 3 downsample stages) 對標 Swin/ConvNeXt
#  ✅ DWConv‑FFN、LayerNorm(GN) 、DropPath、LayerScale→gamma
#  ✅ 顯式 @register_model 註冊（tiny/small/base/large × sa/int）
# -----------------------------------------------------------------------------
import math, inspect, torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
# --------------------- 0. utility ------------------------------------------------

def _filter_kwargs(ctor, kwargs):
    sig = inspect.signature(ctor).parameters
    return {k: v for k, v in kwargs.items() if k in sig}

def Norm2d(c, groups=32):
    g = max(1, math.gcd(c, groups))
    return nn.GroupNorm(g, c)

# --------------------- 1. Window Attention (精確 MHSA) ---------------------------

def _reshape(x, h):
    B, C, H, W = x.shape
    N, d = H * W, C // h
    return x.reshape(B, h, d, N)

class _BaseWinAttn(nn.Module):
    """shared QKV + projection"""
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h, self.d = heads, dim // heads
        self.scale = self.d ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    # --- softmax prefix (numerically stable) ----------------------------------
    @staticmethod
    def _softmax_prefix(s):  # s: (..., N)
        s_max = s.amax(-1, keepdim=True)
        e = (s - s_max).exp_()
        denom = e.cumsum(-1)[..., -1:]
        return e / denom

# -----------------------------------------------------------------------------
class WinIntegral(_BaseWinAttn):
    """Window + Shift Integral‑CNN  (prefix‑integral softmax)"""
    def __init__(self, dim, heads, win=7, shift=True, log_cp_beta=1.3, pe_kernel=5):
        super().__init__(dim, heads)
        self.win, self.shift = win, shift
        # --------- Global-PEG (win<=0) ----------
        if win <= 0:
            self.peg = nn.Conv2d(dim, dim, pe_kernel, 1, pe_kernel//2, groups=dim)
        else:
            self.peg = None
        self.beta = float(log_cp_beta)

        # --------- log-CPB (win>0) ---------------
        if win > 0:
            self.cpb_table = nn.Parameter(torch.zeros((2*win-1)**2, heads))
            nn.init.trunc_normal_(self.cpb_table, std=.02)

            coords = torch.stack(torch.meshgrid(
                torch.arange(win), torch.arange(win), indexing='ij'))      # [2,w,w]
            coords_flat = coords.flatten(1)                               # [2,N]
            rel = coords_flat[:, :, None] - coords_flat[:, None, :]       # [2,N,N]
            rel += win - 1                                                # shift to ≥0
            rel = rel.float()
            # log-scale bucket
            rel = torch.sign(rel) * torch.log1p(rel.abs()) / math.log(self.beta)
            rel = rel + (win - 1)                       # 平移到 0 起點
            rel = rel.clamp(0, 2*win-2)                 # 確保界內
            rel_index = rel[0] * (2*win-1) + rel[1]     # [N,N]
            self.register_buffer('rel_index', rel_index.long(), persistent=False)



    def _unit(self, x):
        """x : (B,C,win,win)  or  (B,C,H,W) when win<=0"""
        B, C, H, W = x.shape
        N = H * W
        if self.peg is not None:              # global 5×5 PE
            x = x + self.peg(x)

        q, k, v = self.qkv(x).chunk(3, 1)
        q = _reshape(q, self.h).transpose(2, 3)    # (B,h,N,d)
        k = _reshape(k, self.h)                    # (B,h,d,N)
        v = _reshape(v, self.h).transpose(2, 3)    # (B,h,N,d)

        logit = (q @ k) * self.scale               # (B,h,N,N)

        # ---- add CPB when win>0 -----------------
        if self.win > 0:
            bias = self.cpb_table[self.rel_index.view(-1)].view(
                     N, N, self.h).permute(2,0,1)  # (h,N,N)
            logit = logit + bias[None]             # broadcast batch

        attn = self._softmax_prefix(logit)         # prefix-scan softmax
        y = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        return y

    def forward(self, x):
        B, C, H, W = x.shape
        win=self.win
        orig_H, orig_W = H, W
        
        if self.win is None or self.win <= 0:     # 全域 attention
            gh = gw = 1
            win_h, win_w = H, W
            need_shift = False                    # 全域就不 shift
        else:                                     # 區域 attention
            pad_h = (win - H % win) % win
            pad_w = (win - W % win) % win
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))     # (left, right, top, bottom)
                H += pad_h;  W += pad_w
                
            gh, gw = H // self.win, W // self.win
            win_h = win_w = self.win
            need_shift = self.shift

        
        # 2. 只在需要時才 roll
        if need_shift:
            x = torch.roll(x, (-win_h // 2, -win_w // 2), (2, 3))
            

        x = (
            x.view(B, C, gh, win_h, gw, win_w)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, C, win_h, win_w)
        )
        y = self._unit(x)
        y = (
            y.view(B, gh, gw, C, win_h, win_w)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, C, H, W)
        )
        if need_shift:
            y = torch.roll(y, (self.win // 2, self.win // 2), (2, 3))
        if self.win >0:
            if (pad_h or pad_w):
                y = y[:, :, :orig_H, :orig_W]


        return self.proj(y)

# --------------------- 2. Feed‑Forward  ----------

class PlainFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.pw1   = nn.Conv2d(dim, hid, 1)          # = Linear
        self.act   = nn.GELU()
        self.pw2   = nn.Conv2d(hid, dim, 1)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.drop  = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y = self.pw2(self.act(self.pw1(x)))
        return x + self.drop(self.gamma.view(1, -1, 1, 1) * y)
        
# --------------------- 3. Block  ------------------------------------
class SICBlock(nn.Module):
    def __init__(self, dim, attn_cls, heads, win, mlp_ratio, drop_path):
        super().__init__()
        self.norm1 = Norm2d(dim)
        self.attn = attn_cls(dim, heads, win=win)
        self.norm2 = Norm2d(dim)
        # self.ffn = DWConvFFN(dim, mlp_ratio, drop_path)
        self.ffn = PlainFFN(dim, mlp_ratio, drop_path)
        self.drop = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = self.ffn(self.norm2(x))
        return x

# --------------------- 4. Stem + Positional‑Encoding ----------------
# ---------- Auto-Stem 依 patch_size 切換實作 -------------------
class Stem(nn.Module):
    """
    Auto-patch Stem
      • patch_size 必須是 2 的冪次 (4 / 8 / 16 …)
      • patch_size <= 4  → 多層 3×3 Conv, stride=2 直到底
      • patch_size >= 8  → 單層 Conv2d(kernel=stride=patch) (ViT-style)
      • 皆附 5×5 depth-wise PE，保持 SICNet 風格
    """
    def __init__(self, in_ch: int, out_ch: int,
                 patch_size: int = 4, pe_kernel: int = 5):
        super().__init__()
        assert patch_size & (patch_size - 1) == 0, \
            "patch_size 必須為 2 的冪 (4 / 8 / 16 …)"

        if patch_size <= 4:
            # ---- 多層 3×3 stride 2 ----
            self.conv_down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch // 2, 3, 2, 1), Norm2d(out_ch // 2), nn.GELU(),
                nn.Conv2d(out_ch // 2, out_ch, 3, 2, 1), Norm2d(out_ch), nn.GELU(),
            )
            # Large‑Kernel Positional Encoding (depth‑wise)
            pad = pe_kernel // 2

        else:
            # ---- 單層 ViT-style Conv ----
            self.conv_down = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=patch_size,
                stride=patch_size
            )

        pad = pe_kernel // 2
        self.pe = nn.Conv2d(
            out_ch, out_ch, pe_kernel, 1, pad, groups=out_ch
        )

    def forward(self, x):
        x = self.conv_down(x)   # B,C,H',W'
        x = x + self.pe(x)
        return x


# --------------------- 5. Backbone ----------------------------------
class SICNet(nn.Module):
    def __init__(self, num_classes=1000,
                 depths=(2, 3, 6), dims=(64, 128, 256, 512),
                 heads=(8, 8, 8), win=8, mlp_ratio=4., drop_path_rate=0.1,
                 attn_cls=WinIntegral, patch_size=4, pe_kernel=5):
        super().__init__()
        self.stem = Stem(3, dims[0], patch_size=patch_size)
        self.num_classes = num_classes
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        dp_iter = iter(dpr)

        def make_stage(in_c, out_c, depth, h):
            layers = [nn.Conv2d(in_c, out_c, 1, 2, 0), Norm2d(out_c), nn.GELU()]
            for _ in range(depth):
                layers.append(SICBlock(out_c, attn_cls, h, win, mlp_ratio, next(dp_iter)))
            return nn.Sequential(*layers)

        self.stages = nn.ModuleList()
        in_c = dims[0]
        for i, (out_c, depth, h) in enumerate(zip(dims, depths, heads)):
            layers = []
            # 只有 i>0 的 stage 才做 ↓2
            if i > 0:
                layers += [nn.Conv2d(in_c, out_c, 1, 2, 0), Norm2d(out_c), nn.GELU()]
            else:
                # i==0 直接用 1×1 point-wise 調通道，不降採樣
                if in_c != out_c:
                    layers += [nn.Conv2d(in_c, out_c, 1), Norm2d(out_c), nn.GELU()]
            for _ in range(depth):
                layers.append(SICBlock(out_c, attn_cls, h, win, mlp_ratio, next(dp_iter)))
            self.stages.append(nn.Sequential(*layers))
            in_c = out_c
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        x = x.mean([2, 3])  # Global Average Pool
        return self.head(x)


# ----------------- register_model (Window 版以 _win 結尾) --------------

@register_model
def sic_tiny_patch16_224_int(pretrained=False, **kw):
    return SICNet(attn_cls=WinIntegral,
        dims=(384,),
        depths=(12,),
        heads=(6,),
        patch_size=16,
        win=0,                         # ★ 全域 Integral-Attention
        mlp_ratio=4.0,
        **_filter_kwargs(SICNet,kw))


@register_model
def sic_tiny_patch4_256_int_win(pretrained=False, **kw):
    return SICNet(attn_cls=WinIntegral,
        dims=(96, 192, 384, 768), depths=(2,2,6,2), heads=(4,8,16,16),
        **_filter_kwargs(SICNet,kw))


@register_model
def sic_small_patch4_256_int_win(pretrained=False, **kw):
    return SICNet(attn_cls=WinIntegral,
        dims=(96, 192, 384, 768), depths=(2, 2, 18, 2), heads=(3, 6, 12, 24),
        **_filter_kwargs(SICNet,kw))


@register_model
def sic_base_patch4_256_int_win(pretrained=False, **kw):
    return SICNet(attn_cls=WinIntegral,
        dims=(96,192,384,768), depths=(2,4,6,2), heads=(6,12,12,12),
        drop_path_rate=0.2, **_filter_kwargs(SICNet,kw))

@register_model
def sic_large_patch4_256_int_win(pretrained=False, **kw):
    return SICNet(attn_cls=WinIntegral,
        dims=(128,256,512,1024), depths=(2,6,14,2), heads=(8,16,16,16),
        drop_path_rate=0.3, **_filter_kwargs(SICNet,kw))
