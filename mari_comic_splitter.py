# -*- coding: utf-8 -*-
"""
ComfyUI Custom Node: Mari Comic Splitter
- Splits a grid image (e.g., 2x3) into individual image files.
- Fix: Auto-increments filename counter to prevent overwriting (e.g., _0001, _0002).
- Fix: Handles RGBA (transparent) images correctly.
- Feature: Supports custom output path.
"""
import torch
import numpy as np
import os
import re
from PIL import Image
import folder_paths

# 1. 초기 매핑 선언
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

class MariComicSplitter:
    def __init__(self):
        self.default_output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "columns": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "filename_prefix": ("STRING", {"default": "Mari_Comic"}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "save_to_disk": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Mari/Comic"

    # ----------------------------------------------------------------
    # 헬퍼 함수: 기존 파일을 스캔해서 다음 번호(Counter)를 찾습니다.
    # 예: prefix_0001.jpg가 있으면 2를 반환
    # ----------------------------------------------------------------
    def _get_next_counter(self, directory, prefix):
        if not os.path.exists(directory):
            return 1
            
        max_counter = 0
        # 정규표현식: prefix_ 뒤에 오는 4자리 숫자를 찾음 (예: Mari_Comic_0005_...)
        # 사용자가 prefix에 특수문자를 썼을 경우를 대비해 escape 처리
        pattern = re.compile(re.escape(prefix) + r"_(\d{4})_")

        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                try:
                    num = int(match.group(1))
                    if num > max_counter:
                        max_counter = num
                except ValueError:
                    continue
        return max_counter + 1

    def run(self, image, rows, columns, filename_prefix, output_path="", save_to_disk=True):
        results = list()
        cropped_tensors = []

        # 1. 저장 경로 결정
        if output_path and output_path.strip() != "":
            target_dir = output_path
        else:
            target_dir = self.default_output_dir

        if save_to_disk and not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
            except Exception as e:
                print(f"[Mari Comic] ⚠️ 폴더 생성 실패, 기본 경로 사용: {e}")
                target_dir = self.default_output_dir

        # 2. 시작 번호(Counter) 계산 (파일 스캔)
        # 이 배치(Batch) 작업이 사용할 시작 번호를 가져옵니다.
        current_counter = self._get_next_counter(target_dir, filename_prefix)

        for batch_idx, img_tensor in enumerate(image):
            # Tensor -> PIL 변환
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 🚑 [RGBA 치료] 투명 배경 -> 흰색 변환
            if img.mode == 'RGBA':
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            slice_width = width // columns
            slice_height = height // rows
            
            # 이번 이미지에 부여할 고유 번호 (배치 내 순서 고려)
            # 만약 배치가 2장이면, 첫 장은 0001, 둘째 장은 0002가 됩니다.
            file_counter = current_counter + batch_idx
            
            cut_index = 1
            for r in range(rows):
                for c in range(columns):
                    left = c * slice_width
                    upper = r * slice_height
                    right = (c + 1) * slice_width
                    lower = (r + 1) * slice_height
                    
                    cropped_img = img.crop((left, upper, right, lower))

                    # 다음 노드용 텐서
                    c_np = np.array(cropped_img).astype(np.float32) / 255.0
                    c_tensor = torch.from_numpy(c_np)[None,]
                    cropped_tensors.append(c_tensor)

                    if save_to_disk:
                        # 3. 파일명 생성 규칙 변경
                        # 형식: {prefix}_{전체카운터(4자리)}_{컷번호(2자리)}.jpg
                        # 예: Mari_Comic_0001_01.jpg, Mari_Comic_0001_02.jpg ...
                        file_name = f"{filename_prefix}_{file_counter:04d}_{cut_index:02d}.jpg"
                        file_path = os.path.join(target_dir, file_name)
                        
                        cropped_img.save(file_path, quality=95)
                        
                        results.append({
                            "filename": file_name,
                            "subfolder": output_path if output_path else "",
                            "type": self.type
                        })
                        print(f"[Mari Comic] Saved: {file_name}")
                    
                    cut_index += 1
        
        if cropped_tensors:
            output_batch = torch.cat(cropped_tensors, dim=0)
        else:
            output_batch = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return {"ui": {"images": results}, "result": (output_batch,)}

# 2. 노드 등록
NODE_CLASS_MAPPINGS.update({
    "Mari Comic Splitter": MariComicSplitter,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "Mari Comic Splitter": "Mari – Comic Grid Splitter",
})