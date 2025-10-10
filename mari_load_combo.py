
# -*- coding: utf-8 -*-
"""
ComfyUI Custom Node: Mari Load Combo (CKPT + LoRA x8 + ver)
- Checkpoint + up to 8 LoRAs
- Master enable switch
- Global LoRA scale knob
- SD version detection for checkpoint and LoRAs
- Output LoRA version list as JSON string
"""

import json
import comfy.sd
import folder_paths

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _ckpt_list():
    try:
        return folder_paths.get_filename_list("checkpoints")
    except Exception:
        return []

def _lora_list():
    try:
        lst = folder_paths.get_filename_list("loras")
    except Exception:
        lst = []
    return ["None"] + lst

def _vae_list():
    try:
        return ["None"] + folder_paths.get_filename_list("vae")
    except Exception:
        return ["None"]

def _guess_sd_version_from_name(name: str) -> str:
    if not name:
        return "Unknown"
    n = name.lower()
    if "sdxl" in n or "xl-" in n or "-xl" in n or "xl.safetensors" in n:
        return "SDXL"
    if "sd2" in n or "2.1" in n or "2.0" in n:
        return "SD 2.x"
    if "sd1" in n or "1.5" in n or "1.4" in n:
        return "SD 1.x"
    return "Unknown"

class MariLoadCombo_CKPT_LoRA:
    @classmethod
    def INPUT_TYPES(cls):
        ckpts = _ckpt_list() or ["None"]
        loras = _lora_list() or ["None"]
        vaes  = _vae_list()  or ["None"]
        def lo(i):
            return {
                f"lora{i}": (loras,),
                f"lora{i}_strength_model": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                f"lora{i}_strength_clip":  ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        lora_inputs = {}
        for i in range(1,9):
            lora_inputs.update(lo(i))
        # Default LoRA1 strength to 1.0
        lora_inputs["lora1_strength_model"] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01})
        lora_inputs["lora1_strength_clip"] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01})

        return {
            "required": {
                "ckpt_name": (ckpts,),
                "enable_loras": ("BOOLEAN", {"default": True}),
                "global_lora_scale": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
                **lora_inputs,
                "guess_config": ("BOOLEAN", {"default": True}),
                "output_vae": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae_name": (vaes,),
            }
        }

    RETURN_TYPES = ("MODEL","CLIP","VAE","STRING","STRING")
    RETURN_NAMES = ("model","clip","vae","ckpt_version","lora_versions")
    FUNCTION = "load_all"
    CATEGORY = "Mari/Loaders"

    def _apply_lora_if_any(self, model, clip, lora_name, sm, sc):
        if lora_name and lora_name != "None" and (abs(sm) > 1e-6 or abs(sc) > 1e-6):
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora_name, sm, sc)
        return model, clip

    def load_all(self, ckpt_name, enable_loras, global_lora_scale,
                 lora1, lora1_strength_model, lora1_strength_clip,
                 lora2, lora2_strength_model, lora2_strength_clip,
                 lora3, lora3_strength_model, lora3_strength_clip,
                 lora4, lora4_strength_model, lora4_strength_clip,
                 lora5, lora5_strength_model, lora5_strength_clip,
                 lora6, lora6_strength_model, lora6_strength_clip,
                 lora7, lora7_strength_model, lora7_strength_clip,
                 lora8, lora8_strength_model, lora8_strength_clip,
                 guess_config, output_vae, vae_name="None"):

        # 1) Load checkpoint
        if guess_config:
            model, clip, vae = comfy.sd.load_checkpoint_guess_config(
                ckpt_name, output_vae=output_vae, vae=vae_name if vae_name != "None" else None
            )
        else:
            model, clip, vae = comfy.sd.load_checkpoint(
                ckpt_name, output_vae=output_vae, vae=vae_name if vae_name != "None" else None
            )

        # 2) LoRA application and version collection
        lora_pairs = [
            (lora1, lora1_strength_model, lora1_strength_clip),
            (lora2, lora2_strength_model, lora2_strength_clip),
            (lora3, lora3_strength_model, lora3_strength_clip),
            (lora4, lora4_strength_model, lora4_strength_clip),
            (lora5, lora5_strength_model, lora5_strength_clip),
            (lora6, lora6_strength_model, lora6_strength_clip),
            (lora7, lora7_strength_model, lora7_strength_clip),
            (lora8, lora8_strength_model, lora8_strength_clip),
        ]

        lora_versions = []
        if enable_loras:
            scale = float(global_lora_scale)
            for lora_name, sm, sc in lora_pairs:
                sm2 = float(sm) * scale
                sc2 = float(sc) * scale
                model, clip = self._apply_lora_if_any(model, clip, lora_name, sm2, sc2)
                if lora_name and lora_name != "None":
                    ver = _guess_sd_version_from_name(lora_name)
                    lora_versions.append({"name": lora_name, "version": ver, "model_strength": sm2, "clip_strength": sc2})
                else:
                    lora_versions.append({"name": None, "version": None})

        # 3) Detect checkpoint version
        ckpt_version = _guess_sd_version_from_name(ckpt_name)

        return (model, clip, vae, ckpt_version, json.dumps(lora_versions))

NODE_CLASS_MAPPINGS.update({
    "Mari Load Combo (CKPT+LoRA x8+ver)": MariLoadCombo_CKPT_LoRA,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "Mari Load Combo (CKPT+LoRA x8+ver)": "Mari – Load Combo",
})
