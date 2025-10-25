import os
from PIL import Image

CATEGORY = "Mari Nodes"

class MariFolderImageScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"multiline": False}),
                "output_folder": ("STRING", {"multiline": False}),
                "target_size": ("INT", {"default": 1600, "min": 1}),
                "interpolation": ([
                    "lanczos",
                    "bilinear",
                    "bicubic",
                    "nearest"
                ],),
            }
        }

    RETURN_TYPES = ("STRING",)  # output_folder 경로를 출력
    RETURN_NAMES = ("output_folder",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, input_folder, output_folder, target_size, interpolation):
        os.makedirs(output_folder, exist_ok=True)
        image_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        interp_map = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST
        }
        interp_mode = interp_map.get(interpolation, Image.LANCZOS)

        total = len(image_files)
        for idx, file_name in enumerate(image_files, start=1):
            in_path = os.path.join(input_folder, file_name)
            out_path = os.path.join(output_folder, file_name)

            img = Image.open(in_path).convert("RGB")
            w, h = img.size

            # 긴쪽 기준 비율로 스케일링
            if w >= h:
                scale = target_size / float(w)
            else:
                scale = target_size / float(h)

            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = img.resize((new_w, new_h), interp_mode)
            resized.save(out_path)

            print(f"[Mari Scale] [{idx}/{total}] {file_name} → {new_w}x{new_h}")

        print(f"[Mari Scale] ✅ 총 {total}장의 이미지를 스케일링 후 저장했습니다.")
        print(f"[Mari Scale] 💾 저장 경로: {output_folder}")

        return (output_folder,)

NODE_CLASS_MAPPINGS = {
    "Mari - Folder Image Scale": MariFolderImageScale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mari - Folder Image Scale": "Mari - Folder Image Scale (Longest Side)"
}
