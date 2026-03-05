import os
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from PIL import Image
from nodes import KSampler, VAEEncode, VAEDecode

CATEGORY = "Mari Nodes"

def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr)[None, ...]  # [1,H,W,3]
    return t

def tensor_to_pil(tensor):
    arr = (tensor[0].cpu().numpy().clip(0,1) * 255).astype(np.uint8)
    return Image.fromarray(arr)

def _parse_int(value, default):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return default
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                print(f"[Mari - Batch Img2Img] ⚠️ Invalid INT '{value}', fallback to {default}")
                return default
    return default

def _parse_float(value, default):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return default
        try:
            return float(v)
        except Exception:
            print(f"[Mari - Batch Img2Img] ⚠️ Invalid FLOAT '{value}', fallback to {default}")
            return default
    return default

def _letterbox_to_canvas(image_tensor, target_h, target_w, padding_color=(0,0,0)):
    b, h, w, c = image_tensor.shape
    if h == target_h and w == target_w:
        return image_tensor

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    nchw = image_tensor.permute(0, 3, 1, 2).contiguous()
    resized = F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    nhwc = resized.permute(0, 2, 3, 1).contiguous()

    canvas = torch.ones((1, target_h, target_w, 3), dtype=torch.float32)
    canvas[..., 0] *= padding_color[0] / 255.0
    canvas[..., 1] *= padding_color[1] / 255.0
    canvas[..., 2] *= padding_color[2] / 255.0

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[:, top:top+new_h, left:left+new_w, :] = nhwc

    return canvas

class MariBatchImg2ImgConditioningSeedMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"multiline": False}),
                "output_folder": ("STRING", {"multiline": False}),
                "steps": ("STRING", {"default": "20"}),
                "cfg": ("STRING", {"default": "7.0"}),
                "denoise_values": ("STRING", {"default": "0.5,0.6,0.7"}),
                "seed": ("STRING", {"default": "-1"}),
                "seed_mode": (["fixed", "per_denoise", "per_image", "per_image_and_denoise", "random"],),
                "mode": (["original", "batch"],),
                "padding_color": ("STRING", {"default": "black"}),
                "sampler_name": ([
                    "euler",
                    "euler_ancestral",
                    "dpmpp_2m",
                    "dpmpp_2m_sde",
                    "heun",
                    "lms",
                    "dpm_2",
                    "dpm_2_ancestral",
                ],),
                "scheduler": ([
                    "normal",
                    "karras",
                    "exponential",
                    "sgm_uniform",
                ],),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def _get_seed(self, base_seed, seed_mode, img_idx, dn_idx):
        if seed_mode == "fixed":
            return base_seed
        elif seed_mode == "per_denoise":
            return base_seed + dn_idx
        elif seed_mode == "per_image":
            return base_seed + img_idx
        elif seed_mode == "per_image_and_denoise":
            return base_seed + img_idx * 1000 + dn_idx  # avoid collisions
        elif seed_mode == "random":
            return random.randint(0, 2**32 - 1)
        else:
            return base_seed

    def run(self, input_folder, output_folder, steps, cfg, denoise_values, seed, seed_mode, mode,
            padding_color, sampler_name, scheduler, model, vae, positive, negative):

        steps_val = max(1, _parse_int(steps, 20))
        cfg_val = _parse_float(cfg, 7.0)
        base_seed = _parse_int(seed, -1)
        if base_seed == -1:
            base_seed = random.randint(0, 2**32 - 1)

        # Padding color parse
        padding_color = padding_color.strip().lower()
        if padding_color == "white":
            pad_color = (255, 255, 255)
        elif padding_color == "black":
            pad_color = (0, 0, 0)
        elif padding_color.startswith("#") and len(padding_color) == 7:
            pad_color = tuple(int(padding_color[i:i+2], 16) for i in (1, 3, 5))
        else:
            pad_color = (0, 0, 0)

        os.makedirs(output_folder, exist_ok=True)

        try:
            denoise_list = [float(v.strip()) for v in denoise_values.split(",") if v.strip()]
        except Exception as e:
            raise ValueError(f"Invalid denoise_values string: '{denoise_values}'.") from e

        image_files = sorted(f for f in os.listdir(input_folder)
                             if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))

        total_tasks = len(image_files) * len(denoise_list)
        if total_tasks == 0:
            print("[Mari - Batch Img2Img] ⚠️ No tasks to run.")
            return ([],)

        output_images = []
        target_h = None
        target_w = None

        sampler = KSampler()
        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()

        start_time = time.time()

        for img_idx, file_name in enumerate(image_files):
            img_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            input_tensor = load_image_tensor(img_path)
            init_latent = vae_encoder.encode(vae, input_tensor)[0]

            for dn_idx, dn in enumerate(denoise_list):
                current_seed = self._get_seed(base_seed, seed_mode, img_idx, dn_idx)

                sampled_latent = sampler.sample(
                    model=model,
                    seed=current_seed,
                    steps=steps_val,
                    cfg=cfg_val,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=init_latent,
                    denoise=dn
                )[0]

                decoded_image = vae_decoder.decode(vae, sampled_latent)[0]

                # batch 모드일 경우 크기 통일
                if mode == "batch":
                    if target_h is None or target_w is None:
                        _, target_h, target_w, _ = decoded_image.shape
                        print(f"[Mari - Batch Img2Img] ▶ Target canvas set to {target_w}x{target_h}.")
                    else:
                        _, h, w, _ = decoded_image.shape
                        if h != target_h or w != target_w:
                            decoded_image = _letterbox_to_canvas(decoded_image, target_h, target_w, pad_color)

                output_images.append(decoded_image)

                save_name = f"{base_name}_denoise{dn}_seed{current_seed}.png"
                save_path = os.path.join(output_folder, save_name)
                tensor_to_pil(decoded_image).save(save_path)

        # If mode is batch, cat them into a single tensor
        if mode == "batch" and len(output_images) > 0:
            output_images = [torch.cat(output_images, dim=0)]

        end_time = time.time()
        total_elapsed = end_time - start_time
        total_min = int(total_elapsed // 60)
        total_sec = int(total_elapsed % 60)
        avg_time = total_elapsed / total_tasks if total_tasks > 0 else 0

        print("=" * 40)
        print("[Mari - Batch Img2Img] ✅ 작업 완료!")
        print(f"  📂 총 파일 수: {len(image_files)}")
        print(f"  🖼️ 생성된 결과물 수: {total_tasks}")
        print(f"  🕒 총 소요 시간: {total_min}m {total_sec}s")
        print(f"  ⏳ 평균 처리 시간: {avg_time:.2f}s/image")
        print(f"  💾 저장 경로: {output_folder}")
        print("=" * 40)

        return (output_images,)

NODE_CLASS_MAPPINGS = {
    "Mari - Batch Img2Img": MariBatchImg2ImgConditioningSeedMode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mari - Batch Img2Img": "Mari - Batch Img2Img"
}
