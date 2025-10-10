
# File: mari_empty_latent_plus.py
# Description: Simplified Empty Latent image node with presets, custom size, percent scale, and batch size.
# Author: Mari
# Category: Mari Nodes/Image
#
# Changes:
# - Inputs reduced to: preset, use_custom, width, height, scale_percent, batch_size
# - Removed: scale_factor, channels, fill_mode, seed, control after generate, snap_multiple, device
# - Always creates zero-filled latents with 4 channels (SD convention)
# - Sizes are snapped internally to multiples of 64 to maintain pipeline compatibility

from typing import Dict, Tuple, List
import torch

# ---- Presets (ordered with a sensible default first) ----
PRESET_ITEMS: List[Tuple[str, Tuple[int, int]]] = [
    ("768 x 768", (768, 768)),
    ("512 x 512", (512, 512)),
    ("640 x 640", (640, 640)),
    ("1024 x 1024", (1024, 1024)),
    ("1280 x 720 (HD)", (1280, 720)),
    ("1920 x 1080 (FHD)", (1920, 1080)),
    ("2048 x 1152 (16:9)", (2048, 1152)),
    ("2560 x 1440 (QHD)", (2560, 1440)),
    ("3840 x 2160 (4K)", (3840, 2160)),
    ("1024 x 576 (16:9)", (1024, 576)),
    ("896 x 512", (896, 512)),
    ("768 x 512", (768, 512)),
    ("1080 x 1920 (9:16)", (1080, 1920)),
    ("768 x 1024", (768, 1024)),
    ("640 x 896", (640, 896)),
    ("512 x 768", (512, 768)),
    ("1350 x 1080 (4:3 IG)", (1350, 1080)),
]

PRESET_TO_SIZE: Dict[str, Tuple[int, int]] = {name: size for name, size in PRESET_ITEMS}
PRESET_NAMES: List[str] = [name for name, _ in PRESET_ITEMS] + ["Custom"]


class MariEmptyLatentPlus:
    """
    Create an empty latent with:
      - resolution presets (dropdown)
      - optional custom width/height
      - percent scaling (100 = no scale)
      - snapped to multiples of 64 internally (no UI control)
    Output: (LATENT, out_width INT, out_height INT)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (PRESET_NAMES,),
                "use_custom": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 64}),
                "scale_percent": ("INT", {"default": 100, "min": 10, "max": 800, "step": 5}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "out_width", "out_height")
    FUNCTION = "create"
    CATEGORY = "Mari Nodes/Image"

    @staticmethod
    def _apply_percent_scale(w: int, h: int, percent: int):
        w = int(round(w * percent / 100.0))
        h = int(round(h * percent / 100.0))
        return max(1, w), max(1, h)

    @staticmethod
    def _snap_size(w: int, h: int, snap: int = 64, min_v: int = 64, max_v: int = 4096):
        def _snap(x):
            x = max(min_v, min(max_v, x))
            q, r = divmod(x, snap)
            if r >= snap / 2:
                q += 1
            return int(q * snap)
        return _snap(w), _snap(h)

    def create(self, preset, use_custom, width, height, scale_percent, batch_size):
        # Resolve base width/height
        if not use_custom and preset in PRESET_TO_SIZE:
            w, h = PRESET_TO_SIZE[preset]
        else:
            w, h = int(width), int(height)

        # Apply percent scaling
        w, h = self._apply_percent_scale(w, h, int(scale_percent))

        # Snap to multiple of 64 for pipeline safety
        w, h = self._snap_size(w, h)

        # Convert to latent spatial size
        lw = max(1, w // 8)
        lh = max(1, h // 8)

        # Always 4-channel, zero-filled latent on CPU (framework will move to device as needed)
        b = int(batch_size)
        samples = torch.zeros(b, 4, lh, lw)

        latent = {"samples": samples}
        return (latent, w, h)


# --- IMPORTANT for aggregator __init__.py ---
NODE_CLASS_MAPPINGS = {
    "MariEmptyLatentPlus": MariEmptyLatentPlus,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MariEmptyLatentPlus": "Mari - Empty Latent+",
}
