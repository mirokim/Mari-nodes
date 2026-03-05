import os
import torch
import numpy as np
from PIL import Image, ImageOps

class SequentialImageLoader:
    # 폴더별 상태 관리: 현재 인덱스와 마지막으로 입력받은 시작값을 저장
    # 구조: { "폴더경로": { "current_idx": 5, "last_start_input": 0 } }
    _state_tracker = {} 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "./ComfyUI/input", "multiline": False}),
                # Reset 버튼 제거하고 시작 인덱스 추가
                "start_at_index": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "filename", "current_index")
    FUNCTION = "load_next_image"
    CATEGORY = "Mari/Image"

    # 매번 실행되도록 설정 (값이 안 바뀌어도 실행되어야 순차적으로 넘어감)
    @classmethod
    def IS_CHANGED(s, directory, start_at_index):
        return float("nan")

    def load_next_image(self, directory, start_at_index):
        # 1. 경로 확인
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"[Mari Nodes] 폴더를 찾을 수 없어: {directory}")

        # 2. 이미지 파일 리스트업
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.gif'}
        files = sorted([
            f for f in os.listdir(directory) 
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

        if not files:
            raise FileNotFoundError(f"[Mari Nodes] 폴더에 이미지가 없어: {directory}")

        # 3. 상태 초기화 및 변경 감지 로직
        # 처음 보는 폴더면 상태를 등록
        if directory not in self._state_tracker:
            self._state_tracker[directory] = {
                "current_idx": start_at_index,
                "last_start_input": start_at_index
            }
        
        state = self._state_tracker[directory]

        # 사용자가 start_at_index 숫자를 바꿨는지 확인!
        # 입력값이 이전과 다르면, 사용자가 "이 번호로 점프해!"라고 명령한 것으로 간주
        if state["last_start_input"] != start_at_index:
            state["current_idx"] = start_at_index
            state["last_start_input"] = start_at_index  # 변경된 값 기억

        # 4. 현재 인덱스 가져오기
        current_idx = state["current_idx"]

        # 5. 파일 선택 (Loop 구조)
        file_to_load = files[current_idx % len(files)]
        
        # 6. 다음 실행을 위해 카운터 +1
        state["current_idx"] += 1

        # 7. 이미지 로딩 및 변환
        image_path = os.path.join(directory, file_to_load)
        
        try:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 256)).convert('L')
            image = i.convert("RGB")
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            print(f"[Mari Nodes] Loading: {file_to_load} (Index: {current_idx})")
            
            return (image, file_to_load, current_idx)
            
        except Exception as e:
            print(f"[Mari Nodes] Error loading image {file_to_load}: {e}")
            raise e

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "Mari_SequentialImageLoader": SequentialImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mari_SequentialImageLoader": "mari - Sequential Image Loader"
}