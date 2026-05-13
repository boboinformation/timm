# timm/models/sicnet_global.py
import torch, torch.nn as nn, inspect
from math import sqrt
from timm.models.registry import register_model

def _filter_kwargs(ctor, kw):          # 同上
    sig = inspect.signature(ctor).parameters
    return {k: v for k, v in kw.items() if k in sig}

_GN_GROUPS=32
def Norm2d(c):
    g=min(_GN_GROUPS,c)
    while c%g: g-=1
    return nn.GroupNorm(g,c)

# ---- attention ---------------------------------------------------------
class _Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); assert dim%heads==0
        self.h,self.d=heads,dim//heads
        self.qkv=nn.Conv2d(dim,dim*3,1,bias=False)
        self.proj=nn.Conv2d(dim,dim,1,bias=False)
        self.scale=1/sqrt(self.d)

    def _split(self,x,B,C,N):
        q,k,v=self.qkv(x).chunk(3,1)
        q=q.reshape(B,self.h,self.d,N).transpose(2,3)
        k=k.reshape(B,self.h,self.d,N)
        v=v.reshape(B,self.h,self.d,N).transpose(2,3)
        return q,k,v

class SAScanAttn(_Attn):
    def forward(self,x):
        B,C,H,W=x.shape;N=H*W
        q,k,v=self._split(x,B,C,N)
        s=(q@k).mul_(self.scale)
        s_max=s.amax(-1,keepdim=True)
        exp=(s-s_max).exp_()
        attn=exp/exp.cumsum(dim=-1)[...,-1:]
        y=(attn@v).transpose(2,3).reshape(B,C,H,W)
        return self.proj(y)

class IntegralAttn(_Attn):
    @staticmethod
    def _prefix(e: torch.Tensor):
        return e.to(dtype=torch.float32).cumsum(dim=-1)
    def forward(self,x):
        B,C,H,W=x.shape;N=H*W
        q,k,v=self._split(x,B,C,N)
        s=(q@k).mul_(self.scale)
        s_max=s.amax(-1,keepdim=True)
        e=(s-s_max).exp_().reshape(-1,N,N)
        denom=self._prefix(e)[...,-1:].reshape(B,self.h,N,1)
        attn=e.reshape(B,self.h,N,N)/denom
        y=(attn@v).transpose(2,3).reshape(B,C,H,W)
        return self.proj(y)

# ---- Block / Net -------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self,dim,attn_cls,heads,mlp_ratio,dp):
        super().__init__()
        self.n1=Norm2d(dim);self.attn=attn_cls(dim,heads)
        self.n2=Norm2d(dim)
        hidden=int(dim*mlp_ratio)
        self.mlp=nn.Sequential(nn.Conv2d(dim,hidden,1),nn.SiLU(),
                               nn.Conv2d(hidden,dim,1))
        self.dp=nn.Dropout(dp) if dp>0 else nn.Identity()
    def forward(self,x):
        x=x+self.dp(self.attn(self.n1(x)))
        x=x+self.dp(self.mlp(self.n2(x)));return x

class Stem(nn.Sequential):
    def __init__(self,in_c,out_c):
        super().__init__(
            nn.Conv2d(in_c,out_c//2,3,2,1),Norm2d(out_c//2),nn.SiLU(),
            nn.Conv2d(out_c//2,out_c,3,2,1),Norm2d(out_c),nn.SiLU())

class SICNet(nn.Module):
    def __init__(self,num_classes=1000,attn_type='sa_scan',
                 dims=(96,192,384,768),depths=(2,4,6),heads=(6,12,12,12),
                 mlp_ratio=4.,drop_path_rate=0.2):
        super().__init__()
        Attn=SAScanAttn if attn_type=='sa_scan' else IntegralAttn
        self.stem=Stem(3,dims[0])
        dpr=torch.linspace(0,drop_path_rate,sum(depths)).tolist()
        cur,in_c=0,dims[0]; stages=[]
        for out_c,d,hd in zip(dims[1:],depths,heads[1:]):
            layers=[nn.Conv2d(in_c,out_c,3,2,1),Norm2d(out_c),nn.SiLU()]
            for j in range(d):
                layers.append(ConvBlock(out_c,Attn,hd,mlp_ratio,dpr[cur+j]))
            cur+=d; in_c=out_c; stages.append(nn.Sequential(*layers))
        self.stages=nn.Sequential(*stages)
        self.head=nn.Linear(dims[-1],num_classes); self.num_classes=num_classes
    def forward(self,x):
        x=self.stem(x);x=self.stages(x);x=x.mean([2,3]);return self.head(x)

# ---- register_model (以 _gbl 結尾) ------------------------------------
@register_model
def sic_tiny_patch16_224_sa_gbl(pretrained=False,**kw):
    return SICNet(attn_type='sa_scan',
        dims=(64,128,256,512),depths=(2,3,6),heads=(8,8,8,8),
        drop_path_rate=0.05,**_filter_kwargs(SICNet,kw))

@register_model
def sic_tiny_patch16_224_int_gbl(pretrained=False,**kw):
    return SICNet(attn_type='integral',
        dims=(96,128,256,512),depths=(2,4,6,2),heads=(4,8,16,16),
        **_filter_kwargs(SICNet,kw))

@register_model
def sic_small_patch16_224_sa_gbl(pretrained=False,**kw):
    return SICNet(attn_type='sa_scan',
        dims=(96,192,384,768),depths=(2,3,9),heads=(6,12,12,12),
        drop_path_rate=0.15,**_filter_kwargs(SICNet,kw))

@register_model
def sic_small_patch16_224_int_gbl(pretrained=False,**kw):
    return SICNet(attn_type='integral',
        dims=(96,192,384,768),depths=(2,3,9),heads=(6,12,12,12),
        drop_path_rate=0.15,**_filter_kwargs(SICNet,kw))

@register_model
def sic_base_patch16_224_sa_gbl(pretrained=False,**kw):
    return SICNet(attn_type='sa_scan',
        dims=(96,192,384,768),depths=(2,4,6),heads=(6,12,12,12),
        drop_path_rate=0.2,**_filter_kwargs(SICNet,kw))

@register_model
def sic_base_patch16_224_int_gbl(pretrained=False,**kw):
    return SICNet(attn_type='integral',
        dims=(96,192,384,768),depths=(2,4,6),heads=(6,12,12,12),
        drop_path_rate=0.2,**_filter_kwargs(SICNet,kw))

@register_model
def sic_large_patch16_224_sa_gbl(pretrained=False,**kw):
    return SICNet(attn_type='sa_scan',
        dims=(128,256,512,1024),depths=(2,6,14),heads=(8,16,16,16),
        drop_path_rate=0.3,**_filter_kwargs(SICNet,kw))

@register_model
def sic_large_patch16_224_int_gbl(pretrained=False,**kw):
    return SICNet(attn_type='integral',
        dims=(128,256,512,1024),depths=(2,6,14),heads=(8,16,16,16),
        drop_path_rate=0.3,**_filter_kwargs(SICNet,kw))
