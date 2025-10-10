
# -*- coding: utf-8 -*-
"""
ComfyUI Custom Node: Mari Color Toolkit
- Global brightness / contrast / saturation / gamma / hue shift
"""
import math
import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _clip01(x): 
    return torch.clamp(x, 0.0, 1.0)

def _adjust_brightness(img, b):
    return _clip01(img * b)

def _adjust_contrast(img, c):
    return _clip01((img - 0.5) * c + 0.5)

def _adjust_saturation(img, s):
    w = torch.tensor([0.299, 0.587, 0.114], dtype=img.dtype, device=img.device)
    l = (img * w.view(1,1,1,3)).sum(dim=-1, keepdim=True)
    return _clip01(l + (img - l) * s)

def _adjust_gamma(img, g):
    # avoid zero
    eps = 1e-8
    return _clip01(torch.pow(torch.clamp(img, eps, 1.0), g))

def _rgb_to_hsv(img):  # (B,H,W,3)
    r,g,b = img[...,0], img[...,1], img[...,2]
    maxc, _ = torch.max(img, dim=-1); minc, _ = torch.min(img, dim=-1)
    v = maxc
    d = maxc - minc + 1e-8
    s = d / (maxc + 1e-8)
    hr = (((g - b) / d) % 6)
    hg = (((b - r) / d) + 2)
    hb = (((r - g) / d) + 4)
    h  = torch.where(maxc==r, hr, torch.where(maxc==g, hg, hb)) / 6.0
    return h.unsqueeze(-1), s.unsqueeze(-1), v.unsqueeze(-1)

def _hsv_to_rgb(h, s, v):
    h = (h % 1.0) * 6.0
    i = torch.floor(h).to(torch.int32)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i0=(i%6==0); i1=(i==1); i2=(i==2); i3=(i==3); i4=(i==4); i5=(i==5)
    r = torch.where(i0, v, torch.where(i1, q, torch.where(i2, p, torch.where(i3, p, torch.where(i4, t, v)))))
    g = torch.where(i0, t, torch.where(i1, v, torch.where(i2, v, torch.where(i3, q, torch.where(i4, p, p)))))
    b = torch.where(i0, p, torch.where(i1, p, torch.where(i2, t, torch.where(i3, v, torch.where(i4, v, q)))))
    return torch.stack([r,g,b], dim=-1)

def _hue_shift(img, degrees):
    if abs(degrees) < 1e-6:
        return img
    h,s,v = _rgb_to_hsv(img)
    h2 = (h + (degrees / 360.0)) % 1.0
    return _hsv_to_rgb(h2, s, v)

class MariColorToolkit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default":1.0, "min":0.0, "max":2.0, "step":0.01}),
                "contrast":   ("FLOAT", {"default":1.0, "min":0.0, "max":2.0, "step":0.01}),
                "saturation": ("FLOAT", {"default":1.0, "min":0.0, "max":2.0, "step":0.01}),
                "gamma":      ("FLOAT", {"default":1.0, "min":0.1, "max":3.0, "step":0.01}),
                "hue_shift":  ("FLOAT", {"default":0.0, "min":-180.0, "max":180.0, "step":1.0}),
            },
            "optional": {
                "global_blend": ("FLOAT", {"default":1.0, "min":0.0, "max":1.0, "step":0.01}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Mari/Color"

    def run(self, image, brightness, contrast, saturation, gamma, hue_shift, global_blend=1.0):
        img = image
        out = img
        if abs(brightness-1.0) > 1e-6: out = _adjust_brightness(out, brightness)
        if abs(saturation-1.0) > 1e-6: out = _adjust_saturation(out, saturation)
        if abs(contrast-1.0)   > 1e-6: out = _adjust_contrast(out, contrast)
        if abs(gamma-1.0)      > 1e-6: out = _adjust_gamma(out, gamma)
        if abs(hue_shift)      > 1e-6: out = _hue_shift(out, hue_shift)
        if global_blend < 1.0:
            out = img * (1.0 - global_blend) + out * global_blend
        return (_clip01(out),)

NODE_CLASS_MAPPINGS.update({
    "Mari Color Toolkit": MariColorToolkit,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "Mari Color Toolkit": "Mari – Color Toolkit",
})
