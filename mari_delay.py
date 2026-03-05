# ComfyUI/custom_nodes/mari_nodes/mari_delay.py
import time

class MariDelayNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delay_seconds": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 300,
                    "step": 1,
                    "display": "number"
                })
            },
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "MODEL", "CLIP", "VAE", "CONDITIONING", "MASK")
    RETURN_NAMES = ("image", "latent", "model", "clip", "vae", "conditioning", "mask")
    FUNCTION = "delay_execution"
    CATEGORY = "Mari"

    def delay_execution(self, delay_seconds, image=None, latent=None, model=None, 
                       clip=None, vae=None, conditioning=None, mask=None):
        print(f"[Mari Delay] Waiting for {delay_seconds} seconds...")
        time.sleep(delay_seconds)
        print("[Mari Delay] Completed!")
        
        return (image, latent, model, clip, vae, conditioning, mask)

NODE_CLASS_MAPPINGS = {
    "Mari - Delay": MariDelayNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mari - Delay": "Mari - Delay"
}