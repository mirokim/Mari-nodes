# __init__.py  (ComfyUI/custom_nodes/mari_nodes/__init__.py)
import os
import re
import glob
import json
from typing import Dict
import torch
import torch.nn.functional as F

# safetensors (optional, for LoRA metadata)
try:
    from safetensors import safe_open
except Exception:
    safe_open = None

# =========================================================
# Shared helpers
# =========================================================
def _ensure_image_tensor(img):
    if not isinstance(img, torch.Tensor):
        raise TypeError("IMAGE must be torch.Tensor [B,H,W,C], float32, 0..1")
    return img

def _clamp01(t: torch.Tensor):
    return torch.clamp(t, 0.0, 1.0)

def _hex_to_rgb(hexstr: str):
    h = hexstr.strip()
    if h.startswith("#"):
        h = h[1:]
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", h):
        raise ValueError(f"Invalid HEX color: {hexstr}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b

# ---- RGB <-> HSV (for saturation/hue) ----
def _rgb_to_hsv(img: torch.Tensor):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc, _ = torch.max(img, dim=-1)
    minc, _ = torch.min(img, dim=-1)
    v = maxc
    delta = maxc - minc

    s = torch.zeros_like(maxc)
    nz = maxc > 0
    s[nz] = delta[nz] / maxc[nz]

    h = torch.zeros_like(maxc)
    mask = delta > 1e-8
    r_eq = (maxc == r) & mask
    g_eq = (maxc == g) & mask
    b_eq = (maxc == b) & mask

    h[r_eq] = (g[r_eq] - b[r_eq]) / delta[r_eq]
    h[g_eq] = 2.0 + (b[g_eq] - r[g_eq]) / delta[g_eq]
    h[b_eq] = 4.0 + (r[b_eq] - g[b_eq]) / delta[b_eq]

    h = (h / 6.0) % 1.0
    return h, s, v

def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor):
    i = torch.floor(h * 6.0).to(torch.int64)
    f = h * 6.0 - i.to(h.dtype)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = torch.where(i_mod == 0, v, torch.where(i_mod == 1, q, torch.where(i_mod == 2, p, torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, v)))))
    g = torch.where(i_mod == 0, t, torch.where(i_mod == 1, v, torch.where(i_mod == 2, v, torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)))))
    b = torch.where(i_mod == 0, p, torch.where(i_mod == 1, p, torch.where(i_mod == 2, t, torch.where(i_mod == 3, v, torch.where(i_mod == 4, v, q)))))

    rgb = torch.stack((r, g, b), dim=-1)
    return _clamp01(rgb)

# ---- resize helpers ----
def _interpolate_nchw(nchw, size_hw, mode):
    align = False if mode in ("bilinear", "bicubic") else None
    try:
        out = F.interpolate(
            nchw, size=size_hw, mode=mode,
            align_corners=align if align is not None else None,
            antialias=True if mode != "nearest" else None
        )
    except TypeError:
        out = F.interpolate(
            nchw, size=size_hw, mode=mode,
            align_corners=align if align is not None else None
        )
    return out

def _align_offsets(canvas_w, canvas_h, img_w, img_h, align):
    ax = (align or "center").lower()
    if "left" in ax: x = 0
    elif "right" in ax: x = canvas_w - img_w
    else: x = (canvas_w - img_w) // 2
    if "top" in ax: y = 0
    elif "bottom" in ax: y = canvas_h - img_h
    else: y = (canvas_h - img_h) // 2
    return int(x), int(y)

# =========================================================
# Node 1: Color Toolkit
# =========================================================
class MariColorToolkit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"mode": (["ColorAdjust", "HexImage"],)},
            "optional": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0,  "max": 1.0,  "step": 0.01}),
                "contrast":   ("FLOAT", {"default": 1.0, "min":  0.0,  "max": 2.0,  "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min":  0.0,  "max": 2.0,  "step": 0.01}),
                "gamma":      ("FLOAT", {"default": 1.0, "min":  0.1,  "max": 5.0,  "step": 0.01}),
                "exposure":   ("FLOAT", {"default": 0.0, "min": -4.0,  "max": 4.0,  "step": 0.01}),
                "hue_shift":  ("FLOAT", {"default": 0.0, "min": -180.0,"max": 180.0,"step": 1.0}),
                "width":     ("INT",   {"default": 1024, "min": 1, "max": 16384}),
                "height":    ("INT",   {"default": 1024, "min": 1, "max": 16384}),
                "hex_color": ("STRING",{"default": "#A14C53"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Mari Nodes/Color"

    def _apply_gamma(self, img, gamma):
        if gamma is None or abs(gamma - 1.0) < 1e-6: return img
        inv = 1.0 / max(1e-6, gamma)
        return _clamp01(torch.pow(_clamp01(img), inv))

    def _apply_exposure(self, img, exposure):
        if exposure is None or abs(exposure) < 1e-6: return img
        return _clamp01(img * float(2.0 ** exposure))

    def _color_adjust(self, image, brightness, contrast, saturation, gamma, exposure, hue_shift_deg):
        img = _ensure_image_tensor(image)
        img = _clamp01(img * contrast + brightness)
        img = self._apply_exposure(img, exposure)
        img = self._apply_gamma(img, gamma)
        h, s, v = _rgb_to_hsv(img)
        s = _clamp01(s * saturation)
        if hue_shift_deg and abs(hue_shift_deg) > 1e-6:
            h = (h + (hue_shift_deg / 360.0)) % 1.0
        return _hsv_to_rgb(h, s, v)

    def _hex_image(self, width, height, hex_color):
        r, g, b = _hex_to_rgb(hex_color)
        arr = torch.ones((1, height, width, 3), dtype=torch.float32)
        arr[..., 0] *= r; arr[..., 1] *= g; arr[..., 2] *= b
        return arr

    def run(self, mode, image=None,
            brightness=0.0, contrast=1.0, saturation=1.0,
            gamma=1.0, exposure=0.0, hue_shift=0.0,
            width=1024, height=1024, hex_color="#A14C53"):
        if mode == "ColorAdjust":
            if image is None: raise ValueError("ColorAdjust mode requires 'image' input.")
            return (self._color_adjust(image, brightness, contrast, saturation, gamma, exposure, hue_shift),)
        elif mode == "HexImage":
            return (self._hex_image(width, height, hex_color),)
        else:
            raise ValueError(f"Unknown mode: {mode}")

# =========================================================
# Node 2: Image Resize
# =========================================================
class MariImageResize:
    SIZE_PRESETS = [
        "Custom",
        "512x512","640x640","768x768","1024x1024","1536x1536","2048x2048","4096x4096",
        "1280x720","1920x1080","2560x1440","3840x2160",
        "720x1280","1080x1920","1440x2560","2160x3840",
        "1024x1536","1536x1024","1200x1800","1800x1200",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["by_size", "by_scale"],),
                "method": (["nearest", "bilinear", "bicubic"],),
            },
            "optional": {
                "size_preset": ((cls.SIZE_PRESETS), {"default": "Custom"}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "keep_aspect": (["stretch", "fit", "fill", "longer_side", "shorter_side"],),
                "align": ([
                    "center","top-left","top","top-right","left","right",
                    "bottom-left","bottom","bottom-right"
                ],),
                "pad_color": ("STRING", {"default": "#000000"}),
                "scale": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01}),
                "ensure_even": ("BOOLEAN", {"default": False}),
                "round_to_multiple": ("INT", {"default": 1, "min": 1, "max": 512}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Mari Nodes/IO"

    def _rounding(self, w, h, ensure_even, mult):
        if ensure_even:
            if w % 2: w += 1
            if h % 2: h += 1
        m = max(1, int(mult))
        if m > 1:
            w = int(round(w / m) * m)
            h = int(round(h / m) * m)
        return max(1, w), max(1, h)

    def _resize_stretch(self, img, out_w, out_h, method):
        nchw = img.permute(0,3,1,2)
        out = _interpolate_nchw(nchw, (out_h, out_w), method).permute(0,2,3,1)
        return _clamp01(out)

    def _resize_fit(self, img, out_w, out_h, method, align, pad_color):
        b, h, w, c = img.shape
        scale = min(out_w / w, out_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = self._resize_stretch(img, new_w, new_h, method)

        pr, pg, pb = _hex_to_rgb(pad_color)
        canvas = torch.ones((b, out_h, out_w, 3), dtype=img.dtype, device=img.device)
        canvas[...,0] *= pr; canvas[...,1] *= pg; canvas[...,2] *= pb

        x, y = _align_offsets(out_w, out_h, new_w, new_h, align or "center")
        canvas[:, y:y+new_h, x:x+new_w, :] = resized
        return _clamp01(canvas)

    def _resize_fill(self, img, out_w, out_h, method, align):
        b, h, w, c = img.shape
        scale = max(out_w / w, out_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = self._resize_stretch(img, new_w, new_h, method)

        x, y = _align_offsets(new_w, new_h, out_w, out_h, align or "center")
        x = max(0, min(x, new_w - out_w))
        y = max(0, min(y, new_h - out_h))
        cropped = resized[:, y:y+out_h, x:x+out_w, :]
        return _clamp01(cropped)

    def _resize_longshort(self, img, out_w, out_h, method, use_longer=True):
        b, h, w, c = img.shape
        if use_longer:
            if w >= h:
                new_w = out_w
                new_h = int(round(h * (out_w / w)))
            else:
                new_h = out_h
                new_w = int(round(w * (out_h / h)))
        else:
            if w <= h:
                new_w = out_w
                new_h = int(round(h * (out_w / w)))
            else:
                new_h = out_h
                new_w = int(round(w * (out_h / h)))
        new_w = max(1, new_w); new_h = max(1, new_h)
        return self._resize_stretch(img, new_w, new_h, method)

    def _apply_preset(self, preset, width, height):
        if preset and preset != "Custom":
            try:
                w, h = preset.lower().split("x")
                return int(w), int(h)
            except Exception:
                pass
        return int(width), int(height)

    def run(self, image, mode, method,
            size_preset="Custom", width=1024, height=1024,
            keep_aspect="stretch", align="center", pad_color="#000000",
            scale=2.0, ensure_even=False, round_to_multiple=1):
        img = _ensure_image_tensor(image)

        if mode == "by_scale":
            b, h, w, c = img.shape
            new_w = max(1, int(round(w * float(scale))))
            new_h = max(1, int(round(h * float(scale))))
            new_w, new_h = self._rounding(new_w, new_h, ensure_even, round_to_multiple)
            return (self._resize_stretch(img, new_w, new_h, method),)

        out_w, out_h = self._apply_preset(size_preset, width, height)
        out_w, out_h = self._rounding(out_w, out_h, ensure_even, round_to_multiple)

        if keep_aspect == "stretch":
            out = self._resize_stretch(img, out_w, out_h, method)
        elif keep_aspect == "fit":
            out = self._resize_fit(img, out_w, out_h, method, align, pad_color)
        elif keep_aspect == "fill":
            out = self._resize_fill(img, out_w, out_h, method, align)
        elif keep_aspect == "longer_side":
            out = self._resize_longshort(img, out_w, out_h, method, use_longer=True)
        elif keep_aspect == "shorter_side":
            out = self._resize_longshort(img, out_w, out_h, method, use_longer=False)
        else:
            raise ValueError(f"Unknown keep_aspect: {keep_aspect}")
        return (out,)

# =========================================================
# Model listing (robust)
# =========================================================
try:
    import folder_paths
except Exception:
    folder_paths = None

MODEL_EXTS = ("*.safetensors","*.ckpt","*.pt","*.bin")

def _scan_dir_for_models(root, sub):
    out = []
    base = os.path.join(root, "models", sub)
    for pat in MODEL_EXTS:
        out += glob.glob(os.path.join(base, pat))
    return sorted(list({os.path.basename(p) for p in out}))

def _list_ckpts():
    try:
        if folder_paths:
            lst = sorted(list(folder_paths.get_filename_list("checkpoints")))
            if lst: return lst
    except Exception:
        pass
    roots = []
    if folder_paths:
        try:
            roots += folder_paths.get_folder_paths("checkpoints_root")
        except Exception:
            pass
    roots += [os.getcwd(), os.path.dirname(os.getcwd())]
    seen = set(); result = []
    for r in roots:
        for name in _scan_dir_for_models(r, "checkpoints"):
            if name not in seen:
                seen.add(name); result.append(name)
    return result

def _list_loras():
    try:
        if folder_paths:
            lst = sorted(list(folder_paths.get_filename_list("loras")))
            if lst: return ["(None)"] + lst
    except Exception:
        pass
    roots = []
    if folder_paths:
        try:
            roots += folder_paths.get_folder_paths("loras_root")
        except Exception:
            pass
    roots += [os.getcwd(), os.path.dirname(os.getcwd())]
    seen = set(); result = []
    for r in roots:
        for name in _scan_dir_for_models(r, "loras"):
            if name not in seen:
                seen.add(name); result.append(name)
    return ["(None)"] + result

def _list_vaes():
    try:
        if folder_paths:
            lst = sorted(list(folder_paths.get_filename_list("vae")))
            if lst: return ["(Default)"] + lst
    except Exception:
        pass
    roots = []
    if folder_paths:
        try:
            roots += folder_paths.get_folder_paths("vae_root")
        except Exception:
            pass
    roots += [os.getcwd(), os.path.dirname(os.getcwd())]
    seen = set(); result = []
    for r in roots:
        for name in _scan_dir_for_models(r, "vae"):
            if name not in seen:
                seen.add(name); result.append(name)
    return ["(Default)"] + result

def _get_full_lora_path(name: str):
    try:
        if folder_paths and name and name != "(None)":
            return folder_paths.get_full_path("loras", name)
    except Exception:
        pass
    return None

def _read_lora_metadata(name: str) -> Dict:
    path = _get_full_lora_path(name)
    if not path or not str(path).lower().endswith(".safetensors") or safe_open is None:
        return {}
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata()
            return dict(meta) if meta else {}
    except Exception:
        return {}

def _infer_lora_version(meta: Dict) -> str:
    cand = (
        meta.get("ss_base_model_version")
        or meta.get("ss_base_model")
        or meta.get("ss_sd_model_name")
        or meta.get("base_model")
        or meta.get("model_name")
        or ""
    )
    if cand:
        return str(cand)
    blob = " ".join([f"{k}:{v}" for k,v in meta.items()]) if meta else ""
    if "sdxl" in blob.lower():
        return "SDXL"
    if any(tag in blob.lower() for tag in ["sd2", "2.x", "v-pred"]):
        return "SD2.x"
    if meta:
        return "unknown"
    return ""

# =========================================================
# Node 3: Load Combo
# =========================================================
try:
    import nodes
except Exception:
    nodes = None

def _detect_family_and_info(model, clip):
    info = {"variant": None, "parameterization": None, "clip_type": None, "notes": []}

    # SDXL check
    try:
        m = getattr(model, "model", model)
        lf = getattr(m, "latent_format", None)
        if lf is not None and "sdxl" in lf.__class__.__name__.lower():
            info["variant"] = "SDXL"
            info["parameterization"] = getattr(m, "parameterization", None)
            if any(hasattr(clip, attr) for attr in ("clip_l", "clip_g")):
                info["clip_type"] = "SDXL Dual-CLIP"
            else:
                info["clip_type"] = "CLIP"
            return "SDXL", info
    except Exception:
        pass

    # Non-SDXL
    family = "SD/SD2"
    try:
        m = getattr(model, "model", model)
        param = getattr(m, "parameterization", None)
        if param is not None:
            info["parameterization"] = str(param)
    except Exception:
        pass

    try:
        ccls = clip.__class__.__name__
        info["clip_type"] = ccls
        if "openclip" in ccls.lower():
            info["notes"].append("OpenCLIP detected")
    except Exception:
        pass

    if info["parameterization"] and info["parameterization"].lower().startswith("v"):
        info["variant"] = "SD2.x"
    else:
        if info["clip_type"] and ("openclip" in info["clip_type"].lower() or "vit_g" in info["clip_type"].lower()):
            info["variant"] = "SD2.x"
        else:
            info["variant"] = "SD1.x"
    return family, info

class MariLoadCombo:
    @classmethod
    def INPUT_TYPES(cls):
        ckpts = _list_ckpts() or ["<no checkpoints found>"]
        loras = _list_loras()
        vaes  = _list_vaes()
        return {
            "required": {"ckpt_name": (ckpts,)},
            "optional": {
                "ckpt_override": ("STRING", {"default": ""}),
                "vae_name":  (vaes,),
                "vae_override": ("STRING", {"default": ""}),
                "clip_skip": ("INT", {"default": 0, "min": 0, "max": 12}),

                "lora1_name": (loras,), "lora1_override": ("STRING", {"default": ""}),
                "lora1_strength_model": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora1_strength_clip":  ("FLOAT", {"default": 0.8, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora1_enabled": ("BOOLEAN", {"default": True}),

                "lora2_name": (loras,), "lora2_override": ("STRING", {"default": ""}),
                "lora2_strength_model": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora2_strength_clip":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora2_enabled": ("BOOLEAN", {"default": False}),

                "lora3_name": (loras,), "lora3_override": ("STRING", {"default": ""}),
                "lora3_strength_model": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora3_strength_clip":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora3_enabled": ("BOOLEAN", {"default": False}),

                "lora4_name": (loras,), "lora4_override": ("STRING", {"default": ""}),
                "lora4_strength_model": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora4_strength_clip":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora4_enabled": ("BOOLEAN", {"default": False}),

                "lora5_name": (loras,), "lora5_override": ("STRING", {"default": ""}),
                "lora5_strength_model": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora5_strength_clip":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora5_enabled": ("BOOLEAN", {"default": False}),

                "lora6_name": (loras,), "lora6_override": ("STRING", {"default": ""}),
                "lora6_strength_model": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora6_strength_clip":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "lora6_enabled": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "family", "info_json")
    FUNCTION = "run"
    CATEGORY = "Mari Nodes/IO"

    def _load_checkpoint(self, ckpt_name):
        if nodes is None:
            raise RuntimeError("ComfyUI 'nodes' module not found.")
        loader = nodes.CheckpointLoaderSimple()
        model, clip, vae = loader.load_checkpoint(ckpt_name)
        return model, clip, vae

    def _apply_clip_skip(self, clip, skip_layers):
        if skip_layers and skip_layers > 0:
            setter = nodes.CLIPSetLastLayer()
            clip = setter.set_last_layer(clip, skip_layers)[0]
        return clip

    def _maybe_override_vae(self, vae, vae_name):
        if vae_name and vae_name != "(Default)":
            vloader = nodes.VAELoader()
            vae = vloader.load_vae(vae_name)[0]
        return vae

    def _apply_lora(self, model, clip, lora_name, sm, sc):
        if not lora_name or lora_name == "(None)":
            return model, clip
        if (sm is None or sm == 0.0) and (sc is None or sc == 0.0):
            return model, clip
        lloader = nodes.LoraLoader()
        model, clip = lloader.load_lora(model, clip, lora_name, float(sm), float(sc))
        return model, clip

    def _or_override(self, dropdown, override):
        ov = (override or "").strip()
        return ov if ov else dropdown

    def run(self,
            ckpt_name, ckpt_override="",
            vae_name="(Default)", vae_override="",
            clip_skip=0,
            lora1_name="(None)", lora1_override="", lora1_strength_model=0.8, lora1_strength_clip=0.8, lora1_enabled=True,
            lora2_name="(None)", lora2_override="", lora2_strength_model=0.0, lora2_strength_clip=0.0, lora2_enabled=False,
            lora3_name="(None)", lora3_override="", lora3_strength_model=0.0, lora3_strength_clip=0.0, lora3_enabled=False,
            lora4_name="(None)", lora4_override="", lora4_strength_model=0.0, lora4_strength_clip=0.0, lora4_enabled=False,
            lora5_name="(None)", lora5_override="", lora5_strength_model=0.0, lora5_strength_clip=0.0, lora5_enabled=False,
            lora6_name="(None)", lora6_override="", lora6_strength_model=0.0, lora6_strength_clip=0.0, lora6_enabled=False,
        ):
        ckpt_final = self._or_override(ckpt_name, ckpt_override)
        vae_final  = self._or_override(vae_name, vae_override)

        model, clip, vae = self._load_checkpoint(ckpt_final)
        clip = self._apply_clip_skip(clip, int(clip_skip))
        vae = self._maybe_override_vae(vae, vae_final)

        slots = [
            (lora1_enabled, self._or_override(lora1_name, lora1_override), lora1_strength_model, lora1_strength_clip),
            (lora2_enabled, self._or_override(lora2_name, lora2_override), lora2_strength_model, lora2_strength_clip),
            (lora3_enabled, self._or_override(lora3_name, lora3_override), lora3_strength_model, lora3_strength_clip),
            (lora4_enabled, self._or_override(lora4_name, lora4_override), lora4_strength_model, lora4_strength_clip),
            (lora5_enabled, self._or_override(lora5_name, lora5_override), lora5_strength_model, lora5_strength_clip),
            (lora6_enabled, self._or_override(lora6_name, lora6_override), lora6_strength_model, lora6_strength_clip),
        ]
        applied_loras = []
        for enabled, name, sm, sc in slots:
            if enabled and name and name != "(None)":
                model, clip = self._apply_lora(model, clip, name, sm, sc)
                meta = _read_lora_metadata(name)
                ver  = _infer_lora_version(meta)
                dim  = meta.get("ss_network_dim") or meta.get("lora_dim")
                alpha= meta.get("ss_network_alpha")
                applied_loras.append({
                    "name": name, "version": ver, "dim": dim, "alpha": alpha,
                    "strength_model": float(sm), "strength_clip": float(sc)
                })

        family, info = _detect_family_and_info(model, clip)
        info.update({
            "ckpt_name": ckpt_final,
            "vae_name": vae_final or "(Default)",
            "clip_skip": int(clip_skip),
            "applied_loras": applied_loras,
        })
        info_json = json.dumps(info, ensure_ascii=False)
        return (model, clip, vae, family, info_json)

# =========================================================
# Registration
# =========================================================
NODE_CLASS_MAPPINGS = {
    "MariColorToolkit": MariColorToolkit,
    "MariImageResize": MariImageResize,
    "MariLoadCombo": MariLoadCombo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MariColorToolkit": "Mari Nodes - Color Toolkit",
    "MariImageResize": "Mari Nodes - Image Resize",
    "MariLoadCombo": "Mari Nodes - Load Combo",
}
print("[Mari Nodes] registered: Color Toolkit, Image Resize, Load Combo")
