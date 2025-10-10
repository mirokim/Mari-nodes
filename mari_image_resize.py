
# -*- coding: utf-8 -*-
"""
ComfyUI Custom Node: Mari Image Resize
- Final node name: "Mari Image Resize"
- Two modes:
  1) scale  : resize by scale factor(s)
      - `scale` (single-number uniform scale). If not 1.0, overrides scale_x/scale_y.
      - scale_x/scale_y + lock_aspect for advanced use.
  2) custom : resize to explicit width/height (stretch)
- Interpolation: nearest, bilinear, bicubic, area (downsample)
- Inputs: IMAGE (required), MASK (optional)
- Outputs: image, mask, info
"""
import torch
import torch.nn.functional as F

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _clip01(x): return torch.clamp(x, 0.0, 1.0)
def _to_bchw(img_bhwc): return img_bhwc.permute(0,3,1,2)
def _to_bhwc(img_bchw): return img_bchw.permute(0,2,3,1)

def _resize_bchw(x, size, mode):
    if mode == "area":
        in_h, in_w = x.shape[-2:]
        if size[0] >= in_h or size[1] >= in_w:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return F.interpolate(x, size=size, mode="area")
    if mode in ("bilinear","bicubic"):
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)

def _resize_mask_like(mask, ref_img):
    if mask is None:
        return torch.zeros((ref_img.shape[0], ref_img.shape[1], ref_img.shape[2], 1), dtype=ref_img.dtype, device=ref_img.device)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        m = mask.permute(0,3,1,2)
    elif mask.ndim == 3:
        m = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        m = mask
    else:
        raise ValueError("MASK tensor shape not supported")
    m = _resize_bchw(m, (ref_img.shape[1], ref_img.shape[2]), mode="bilinear")
    return torch.clamp(m.permute(0,2,3,1), 0.0, 1.0)

class MariImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["scale","custom"],),
                "method": (["bilinear","bicubic","nearest","area"],),
                # --- SIMPLE UNIFORM SCALE ---
                "scale": ("FLOAT", {"default":1.0, "min":0.01, "max":10.0, "step":0.01}),
                # --- ADVANCED SCALE ---
                "scale_x": ("FLOAT", {"default":1.0, "min":0.01, "max":10.0, "step":0.01}),
                "scale_y": ("FLOAT", {"default":1.0, "min":0.01, "max":10.0, "step":0.01}),
                "lock_aspect": ("BOOLEAN", {"default": True}),
                # --- CUSTOM SIZE ---
                "target_width": ("INT", {"default":1024, "min":8, "max":8192, "step":8}),
                "target_height": ("INT", {"default":1024, "min":8, "max":8192, "step":8}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE","MASK","STRING")
    RETURN_NAMES = ("image","mask","info")
    FUNCTION = "run"
    CATEGORY = "Mari/Image"

    def run(self, image, mode, method, scale, scale_x, scale_y, lock_aspect, target_width, target_height, mask=None):
        img = image  # (B,H,W,C)
        b,h,w,c = img.shape
        info = ""

        if mode == "scale":
            # If simple scale != 1.0, use it for both axes
            if abs(scale - 1.0) > 1e-6:
                sx = sy = float(scale)
                reason = "simple scale"
            else:
                sx = float(scale_x)
                sy = float(scale_y if not lock_aspect else scale_x)
                reason = f"advanced scale (lock_aspect={bool(lock_aspect)})"
            new_w = max(1, int(round(w * sx)))
            new_h = max(1, int(round(h * sy)))
            out_img = _to_bhwc(_resize_bchw(_to_bchw(img), (new_h, new_w), method))
            info = f"{reason}: x={sx:.4f}, y={sy:.4f} -> ({new_w}x{new_h})"
        elif mode == "custom":
            W = int(target_width); H = int(target_height)
            out_img = _to_bhwc(_resize_bchw(_to_bchw(img), (H, W), method))
            info = f"custom size -> ({W}x{H})"
        else:
            raise ValueError("Unknown mode")

        out_mask = _resize_mask_like(mask, out_img)
        return (_clip01(out_img), _clip01(out_mask), info)

NODE_CLASS_MAPPINGS.update({
    "Mari Image Resize": MariImageResize,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "Mari Image Resize": "Mari – Image Resize",
})
