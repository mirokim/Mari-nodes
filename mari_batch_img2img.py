import os
import time
import torch
import random
import numpy as np
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
    # Accept int/float/str (possibly empty), fallback to default
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

class MariBatchImg2ImgConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        # NOTE: steps/cfg are STRING to avoid UI validation errors when empty strings are present.
        return {
            "required": {
                "input_folder": ("STRING", {"multiline": False}),
                "output_folder": ("STRING", {"multiline": False}),
                "steps": ("STRING", {"default": "20"}),   # robust
                "cfg": ("STRING", {"default": "7.0"}),    # robust
                "denoise_values": ("STRING", {"default": "0.5,0.6,0.7"}),
                "seed": ("STRING", {"default": "-1"}),    # robust: allow empty / string
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
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, input_folder, output_folder, steps, cfg,
            denoise_values, seed, sampler_name, scheduler, model, vae, positive, negative):

        # Robust parsing
        steps_val = max(1, _parse_int(steps, 20))
        cfg_val = _parse_float(cfg, 7.0)
        seed_val = _parse_int(seed, -1)

        os.makedirs(output_folder, exist_ok=True)

        try:
            denoise_list = [float(v.strip()) for v in denoise_values.split(",") if v.strip()]
        except Exception as e:
            raise ValueError(f"Invalid denoise_values string: '{denoise_values}'. Use comma-separated floats like '0.5,0.6,0.7'.") from e

        image_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        total_tasks = len(image_files) * len(denoise_list)
        if total_tasks == 0:
            print("[Mari - Batch Img2Img] ⚠️ No tasks to run. Check input folder or denoise_values.")
            return (torch.empty(0),)

        task_index = 0
        all_outputs = []

        base_seed = random.randint(0, 2**32 - 1) if seed_val == -1 else seed_val

        start_time = time.time()
        sampler = KSampler()
        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()

        for file_index, file_name in enumerate(image_files):
            img_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            input_tensor = load_image_tensor(img_path)

            init_latent = vae_encoder.encode(vae, input_tensor)[0]

            for dn_index, dn in enumerate(denoise_list):
                task_index += 1
                current_seed = base_seed + dn_index  # seed increment per denoise

                elapsed = time.time() - start_time
                avg_time = elapsed / task_index if task_index > 0 else 0
                eta = avg_time * (total_tasks - task_index)
                eta_min = int(eta // 60)
                eta_sec = int(eta % 60)

                print(f"[Mari - Batch Img2Img] [{task_index}/{total_tasks}]")
                print(f"  ➝ File: {file_name}")
                print(f"  ➝ Denoise: {dn}")
                print(f"  ➝ Seed: {current_seed}")
                print(f"  ➝ Sampler: {sampler_name}, Scheduler: {scheduler}")
                print(f"  ➝ Steps: {steps_val}, CFG: {cfg_val}")
                print(f"  ⏳ ETA: {eta_min}m {eta_sec}s (avg {avg_time:.2f}s/image)")
                print("-" * 40)

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
                all_outputs.append(decoded_image)

                save_name = f"{base_name}_denoise{dn}_seed{current_seed}.png"
                save_path = os.path.join(output_folder, save_name)
                tensor_to_pil(decoded_image).save(save_path)

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

        return (torch.cat(all_outputs, dim=0),)

NODE_CLASS_MAPPINGS = {
    "Mari - Batch Img2Img": MariBatchImg2ImgConditioning
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mari - Batch Img2Img": "Mari - Batch Img2Img"
}
