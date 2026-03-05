# -*- coding: utf-8 -*-
"""
ComfyUI Custom Node: Mari Video Frame Extractor
- Extract frames from video at specified intervals
- Save to output folder
- Return frames as IMAGE batch for further processing
"""

import os
import cv2
import torch
import numpy as np
import folder_paths

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

class MariVideoFrameExtractor:
    """
    Extract frames from video file at regular intervals.
    Saves extracted frames to output folder and returns as IMAGE batch.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {
                    "default": "video_frames",
                    "multiline": False
                }),
                "frame_interval": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "label": "프레임 간격 (Frame Interval)"
                }),
                "filename_prefix": ("STRING", {
                    "default": "frame",
                    "multiline": False
                }),
                "save_format": (["png", "jpg", "webp"], {
                    "default": "png"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "label": "시작 프레임 (Start Frame)"
                }),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "label": "최대 프레임 수 (0=무제한)"
                }),
            },
            "optional": {
                "images": ("IMAGE",),  # VHS/AnimateDiff 등의 IMAGE 배치 출력
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": True  # 노드 연결만 가능
                }),
                "video_path_text": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }
    
    INPUT_IS_LIST = False
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "output_path", "frame_count")
    FUNCTION = "extract_frames"
    CATEGORY = "Mari/Video"

    def extract_frames(self, output_folder, frame_interval, 
                      filename_prefix, save_format, start_frame, max_frames,
                      images=None, video_path=None, video_path_text=""):
        
        # Setup output directory
        output_dir = folder_paths.get_output_directory()
        full_output_path = os.path.join(output_dir, output_folder)
        os.makedirs(full_output_path, exist_ok=True)
        
        frames_list = []
        saved_count = 0
        
        # Case 1: images (VIDEO 노드에서 IMAGE 텐서를 받은 경우)
        if images is not None:
            print("=" * 60)
            print(f"[Mari Video Extractor] 📹 비디오 텐서 입력 감지")
            print(f"  📦 텐서 크기: {images.shape}")
            print(f"  🔢 추출 간격: {frame_interval}")
            print("=" * 60)
            
            total_frames = images.shape[0]
            
            # 프레임 간격에 따라 샘플링
            for frame_idx in range(start_frame, total_frames, frame_interval):
                if max_frames > 0 and saved_count >= max_frames:
                    print(f"[Mari Video Extractor] ⚠️ 최대 프레임 수({max_frames})에 도달했습니다.")
                    break
                
                frame_tensor = images[frame_idx]  # [H, W, C]
                
                # 파일로 저장
                ext = save_format
                filename = f"{filename_prefix}_{saved_count:05d}.{ext}"
                filepath = os.path.join(full_output_path, filename)
                
                # Tensor to numpy for saving
                frame_np = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                
                if ext == "png":
                    cv2.imwrite(filepath, frame_bgr)
                elif ext == "jpg":
                    cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                elif ext == "webp":
                    cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_WEBP_QUALITY, 95])
                
                frames_list.append(frame_tensor)
                saved_count += 1
                
                if saved_count % 10 == 0:
                    print(f"[Mari Video Extractor] 📸 저장 중... {saved_count}개 프레임")
            
            images_batch = torch.stack(frames_list, dim=0)
        
        # Case 2: video_path (노드 연결) 또는 video_path_text (직접 입력)
        else:
            # 우선순위: video_path (노드 연결) > video_path_text (직접 입력)
            path = None
            if video_path:
                path = video_path
            elif video_path_text and video_path_text.strip():
                path = video_path_text.strip()
            
            if not path:
                raise ValueError("❌ [Mari Video Extractor] 비디오 입력(images) 또는 비디오 경로를 제공해주세요.")
            
            if not os.path.exists(path):
                raise ValueError(f"❌ [Mari Video Extractor] 비디오 파일을 찾을 수 없습니다: {path}")
            
            cap = cv2.VideoCapture(path)
            
            if not cap.isOpened():
                raise ValueError(f"❌ [Mari Video Extractor] 비디오 파일을 열 수 없습니다: {path}")
            
            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print("=" * 60)
            print(f"[Mari Video Extractor] 📹 비디오 정보")
            print(f"  📁 파일: {os.path.basename(path)}")
            print(f"  📏 해상도: {video_width}x{video_height}")
            print(f"  🎬 전체 프레임: {total_frames}")
            print(f"  ⏱️  FPS: {fps:.2f}")
            print(f"  ⏯️  시작 프레임: {start_frame}")
            print(f"  🔢 추출 간격: {frame_interval}")
            print("=" * 60)
            
            frame_count = 0
            
            # Set starting position
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_count = start_frame
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check if we should extract this frame
                if (frame_count - start_frame) % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Save to file
                    ext = save_format
                    filename = f"{filename_prefix}_{saved_count:05d}.{ext}"
                    filepath = os.path.join(full_output_path, filename)
                    
                    if ext == "png":
                        cv2.imwrite(filepath, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    elif ext == "jpg":
                        cv2.imwrite(filepath, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), 
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                    elif ext == "webp":
                        cv2.imwrite(filepath, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_WEBP_QUALITY, 95])
                    
                    # Convert to ComfyUI IMAGE format (float32, 0-1 range)
                    image_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
                    frames_list.append(image_tensor)
                    
                    saved_count += 1
                    
                    # Print progress every 10 frames
                    if saved_count % 10 == 0:
                        print(f"[Mari Video Extractor] 📸 추출 중... {saved_count}개 프레임")
                    
                    # Check max_frames limit
                    if max_frames > 0 and saved_count >= max_frames:
                        print(f"[Mari Video Extractor] ⚠️ 최대 프레임 수({max_frames})에 도달했습니다.")
                        break
                
                frame_count += 1
            
            cap.release()
            
            if len(frames_list) == 0:
                raise ValueError("❌ [Mari Video Extractor] 추출된 프레임이 없습니다.")
            
            # Stack frames into batch tensor [B, H, W, C]
            images_batch = torch.stack(frames_list, dim=0)
        
        print("=" * 60)
        print(f"[Mari Video Extractor] ✅ 추출 완료!")
        print(f"  🖼️  추출된 프레임 수: {saved_count}")
        print(f"  💾 저장 경로: {full_output_path}")
        print(f"  📦 배치 텐서 크기: {images_batch.shape}")
        print("=" * 60)
        
        return (images_batch, full_output_path, saved_count)


NODE_CLASS_MAPPINGS.update({
    "Mari Video Frame Extractor": MariVideoFrameExtractor,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "Mari Video Frame Extractor": "Mari – Video Frame Extractor",
})
