# -*- coding: utf-8 -*-
"""
ComfyUI Custom Node: Mari Smart Resize (Conditional + Axis Control)
- Checks if image is larger than trigger_size.
- Resize Modes:
    1. Longest Side: Fits the longest side to target pixels.
    2. Width: Fits width to target pixels (height adjusts auto).
    3. Height: Fits height to target pixels (width adjusts auto).
    4. Percentage: Scales by percentage.
"""
import torch
import comfy.utils

class MariSmartResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                
                # 1. 작동 조건: 이 크기보다 크면 작동
                "trigger_size": ("INT", {
                    "default": 2500, 
                    "min": 0, 
                    "max": 32768, 
                    "step": 1, 
                    "label": "감지 기준 크기 (Trigger)" 
                }),

                # 2. 리사이즈 기준 선택 (가로 / 세로 / 긴 변 / 퍼센트)
                "resize_mode": ([
                    "based_on_longest",  # 긴 변 기준 (기존 방식)
                    "based_on_width",    # 가로 길이 기준 (New!)
                    "based_on_height",   # 세로 길이 기준 (New!)
                    "by_percentage"      # 퍼센트 비율
                ], {
                    "default": "based_on_longest", 
                    "label": "리사이즈 기준 (Mode)"
                }),

                # 3. 픽셀 단위 목표값 (가로/세로/긴변 모드용)
                "target_pixels": ("INT", {
                    "default": 1024, 
                    "min": 1, 
                    "max": 32768, 
                    "step": 1, 
                    "label": "목표 픽셀값 (Target Pixels)" 
                }),

                # 4. 퍼센트 단위 목표값 (퍼센트 모드용)
                "target_percent": ("INT", {
                    "default": 50, 
                    "min": 1, 
                    "max": 200, 
                    "step": 1, 
                    "label": "목표 비율 % (Percent)" 
                }),

                "method": (["lanczos", "nearest-exact", "bilinear", "area", "bicubic"], {"default": "lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "resize_if_needed"
    CATEGORY = "Mari/Image"

    def resize_if_needed(self, image, trigger_size, resize_mode, target_pixels, target_percent, method):
        # image shape: [Batch, Height, Width, Channel]
        _, h, w, _ = image.shape
        
        # 1. 조건 확인: 이미지가 trigger_size보다 큰가?
        if w >= trigger_size or h >= trigger_size:
            
            scale_factor = 1.0
            mode_desc = ""

            # 2. 모드별 배율 계산
            if resize_mode == "based_on_longest":
                # 기존 방식: 긴 쪽을 맞춤
                scale_factor = target_pixels / max(w, h)
                mode_desc = "Longest Side"
                
            elif resize_mode == "based_on_width":
                # 가로 고정 (세로는 비율따라 자동)
                scale_factor = target_pixels / w
                mode_desc = "Width Fixed"
                
            elif resize_mode == "based_on_height":
                # 세로 고정 (가로는 비율따라 자동)
                scale_factor = target_pixels / h
                mode_desc = "Height Fixed"

            elif resize_mode == "by_percentage":
                # 퍼센트
                scale_factor = target_percent / 100.0
                mode_desc = f"Percentage ({target_percent}%)"

            # 3. 새로운 크기 계산
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            reason = f"Activated({mode_desc}): {w}x{h} → {new_w}x{new_h}"
            print(f"[Mari Smart Resize] 📏 {reason}")
            
            # 4. 리사이즈 실행
            # ComfyUI image format [B, H, W, C] -> common_upscale expects [B, C, H, W]
            samples = image.movedim(-1, 1)
            s = comfy.utils.common_upscale(samples, new_w, new_h, method, "disabled")
            s = s.movedim(1, -1)
            
            return (s, reason)
        
        else:
            # 조건 미달: 그냥 통과
            reason = f"Pass: ({w}x{h}) < Trigger({trigger_size})"
            print(f"[Mari Smart Resize] ✅ {reason}")
            return (image, reason)

# Node Registry
NODE_CLASS_MAPPINGS = {
    "MariSmartResize": MariSmartResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MariSmartResize": "Mari – Smart Resize (Advanced)"
}