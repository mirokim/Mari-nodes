# ComfyUI/custom_nodes/mari_nodes/mari_clip_autosave.py
"""
Mari CLIP Text Encode (Auto Save)
- CLIPTextEncode 기능 + prompt 변경 시 자동 저장
- JSON 파일에 날짜/시간과 함께 prompt 히스토리를 기록
"""

import os
import json
import datetime


class MariClipAutoSave:
    """
    기본 CLIPTextEncode 기능을 확장하여,
    prompt 텍스트가 변경될 때마다 자동으로 파일에 저장합니다.
    """

    # 마지막으로 저장한 prompt를 기억 (중복 저장 방지)
    _last_prompt = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "인코딩할 텍스트 프롬프트"
                }),
                "clip": ("CLIP", {
                    "tooltip": "텍스트 인코딩에 사용할 CLIP 모델"
                }),
            },
            "optional": {
                "save_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "저장 경로 (비어있으면 output/mari_prompt_logs 에 저장)"
                }),
                "filename": ("STRING", {
                    "default": "prompt_history",
                    "multiline": False,
                    "tooltip": "저장 파일명 (확장자 제외)"
                }),
                "save_format": (["json", "txt"],),
                "label": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "프롬프트에 붙일 라벨/메모 (선택)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING",)
    RETURN_NAMES = ("conditioning", "saved_path",)
    FUNCTION = "encode_and_save"
    CATEGORY = "Mari/Conditioning"
    DESCRIPTION = "CLIP Text Encode + 프롬프트 자동 저장. 텍스트가 변경될 때마다 날짜/시간과 함께 기록합니다."

    @classmethod
    def IS_CHANGED(cls, text, clip, save_dir="", filename="prompt_history",
                   save_format="json", label=""):
        # prompt 내용이 바뀔 때마다 실행되도록 함
        return text

    def _get_save_dir(self, save_dir):
        """저장 디렉토리 결정"""
        if save_dir and save_dir.strip():
            path = save_dir.strip()
        else:
            # ComfyUI output 폴더 하위에 기본 경로 생성
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except Exception:
                output_dir = os.path.join(os.getcwd(), "output")
            path = os.path.join(output_dir, "mari_prompt_logs")

        os.makedirs(path, exist_ok=True)
        return path

    def _save_as_json(self, filepath, text, label):
        """JSON 형식으로 프롬프트 히스토리 저장"""
        now = datetime.datetime.now()
        entry = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "prompt": text,
        }
        if label and label.strip():
            entry["label"] = label.strip()

        # 기존 파일이 있으면 로드 후 추가
        data = {"prompt_history": []}
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "prompt_history" not in data:
                    data["prompt_history"] = []
            except (json.JSONDecodeError, IOError):
                data = {"prompt_history": []}

        data["prompt_history"].append(entry)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_as_txt(self, filepath, text, label):
        """TXT 형식으로 프롬프트 히스토리 저장"""
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append(f"[{timestamp}]")
        if label and label.strip():
            lines.append(f"Label: {label.strip()}")
        lines.append(text)
        lines.append("-" * 60)
        lines.append("")

        with open(filepath, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def encode_and_save(self, text, clip, save_dir="", filename="prompt_history",
                        save_format="json", label=""):
        # --- 1) CLIP 인코딩 (기본 CLIPTextEncode 동작) ---
        if clip is None:
            raise RuntimeError(
                "❌ [Mari CLIP AutoSave] clip 입력이 없습니다 (None).\n"
                "체크포인트 로더에서 유효한 CLIP 모델이 로드되었는지 확인하세요."
            )

        tokens = clip.tokenize(text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # --- 2) 프롬프트 자동 저장 ---
        node_id = id(self)
        last = MariClipAutoSave._last_prompt.get(node_id)

        if text != last:
            # 저장 경로 설정
            dir_path = self._get_save_dir(save_dir)
            safe_filename = filename.strip() if filename and filename.strip() else "prompt_history"
            ext = ".json" if save_format == "json" else ".txt"
            filepath = os.path.join(dir_path, safe_filename + ext)

            # 저장
            if save_format == "json":
                self._save_as_json(filepath, text, label)
            else:
                self._save_as_txt(filepath, text, label)

            MariClipAutoSave._last_prompt[node_id] = text
            print(f"[Mari CLIP AutoSave] ✅ 프롬프트 저장 완료: {filepath}")
        else:
            dir_path = self._get_save_dir(save_dir)
            safe_filename = filename.strip() if filename and filename.strip() else "prompt_history"
            ext = ".json" if save_format == "json" else ".txt"
            filepath = os.path.join(dir_path, safe_filename + ext)
            print(f"[Mari CLIP AutoSave] ⏭️ 동일 프롬프트 - 저장 건너뜀")

        return (conditioning, filepath,)


NODE_CLASS_MAPPINGS = {
    "Mari - CLIP AutoSave": MariClipAutoSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mari - CLIP AutoSave": "Mari - CLIP Text Encode (Auto Save)",
}
