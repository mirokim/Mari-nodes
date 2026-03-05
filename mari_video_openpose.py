# -*- coding: utf-8 -*-
"""
Mari Video OpenPose Extractor for ComfyUI
동영상에서 OpenPose를 추출하여 동영상으로 출력하는 노드
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import time

CATEGORY = "Mari Nodes"

try:
    from controlnet_aux import OpenposeDetector
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    print("[Mari - Video OpenPose] ⚠️ controlnet_aux not found. Install with: pip install controlnet-aux")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[Mari - Video OpenPose] ⚠️ mediapipe not found. Install with: pip install mediapipe")


class VideoWrapper:
    """ComfyUI VIDEO 타입 완전 호환 래퍼"""
    def __init__(self, video_path):
        self.video = video_path
        self.path = video_path
        self.file = video_path
        self.filepath = video_path
        self.filename = os.path.basename(video_path) if video_path else ""
        
        self._cap = None
        
        # 동영상 정보 로드
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = self.frame_count / self.fps if self.fps > 0 else 0
                cap.release()
            else:
                self.width = 1920
                self.height = 1080
                self.fps = 30.0
                self.frame_count = 0
                self.duration = 0
        else:
            self.width = 1920
            self.height = 1080
            self.fps = 30.0
            self.frame_count = 0
            self.duration = 0
    
    # 기본 속성 접근 메서드
    def get_dimensions(self):
        """동영상 해상도 반환"""
        return (self.width, self.height)
    
    def get_fps(self):
        """FPS 반환"""
        return self.fps
    
    def get_frame_count(self):
        """총 프레임 수 반환"""
        return self.frame_count
    
    def get_path(self):
        """경로 반환"""
        return self.video
    
    def get_file(self):
        """파일 경로 반환"""
        return self.video
    
    def get_duration(self):
        """재생 시간 반환 (초)"""
        return self.duration
    
    # 파일 저장 메서드
    def save_to(self, destination_path):
        """동영상을 다른 위치에 저장"""
        import shutil
        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy2(self.video, destination_path)
            print(f"[VideoWrapper] 💾 Video saved to: {destination_path}")
            return destination_path
        except Exception as e:
            print(f"[VideoWrapper] ❌ Error saving video: {e}")
            raise
    
    def copy_to(self, destination_path):
        """save_to의 별칭"""
        return self.save_to(destination_path)
    
    # VideoCapture 관련 메서드
    def load(self):
        """VideoCapture 객체 생성"""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.video)
        return self._cap
    
    def read(self):
        """프레임 읽기"""
        if self._cap is None:
            self.load()
        return self._cap.read()
    
    def release(self):
        """VideoCapture 해제"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    # Context manager 지원
    def __enter__(self):
        """with 문 지원"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 문 종료 시 자동 해제"""
        self.release()
        return False
    
    # 문자열 표현
    def __str__(self):
        return self.video
    
    def __repr__(self):
        return f"VideoWrapper('{self.video}', {self.width}x{self.height}, {self.fps}fps, {self.frame_count} frames)"
    
    # 딕셔너리처럼 접근 가능하게
    def __getitem__(self, key):
        """딕셔너리 스타일 접근"""
        if key in ['video', 'path', 'file', 'filepath']:
            return self.video
        elif key == 'width':
            return self.width
        elif key == 'height':
            return self.height
        elif key == 'fps':
            return self.fps
        elif key == 'frame_count':
            return self.frame_count
        else:
            raise KeyError(f"Key '{key}' not found")
    
    def get(self, key, default=None):
        """딕셔너리 스타일 get"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self):
        """딕셔너리 스타일 keys"""
        return ['video', 'path', 'file', 'filepath', 'width', 'height', 'fps', 'frame_count']
    
    # 속성 동적 접근
    def __getattr__(self, name):
        """알 수 없는 속성 요청 시 video 경로 반환 (안전장치)"""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # 알려지지 않은 속성은 video 경로 반환 (호환성)
        print(f"[VideoWrapper] ⚠️ Unknown attribute '{name}' requested, returning video path")
        return self.video


class MariVideoOpenPoseExtractor:
    """
    동영상에서 OpenPose를 추출하여 동영상으로 출력하는 노드
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        methods = []
        if OPENPOSE_AVAILABLE:
            methods.append("controlnet_openpose")
        if MEDIAPIPE_AVAILABLE:
            methods.append("mediapipe")
        
        if not methods:
            methods = ["none_available"]
        
        return {
            "required": {
                "video": ("VIDEO",),  # VIDEO input 노드로 받기
                "detection_method": (methods,),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1
                }),
                "max_frames": ("INT", {
                    "default": 0,  # 0 = all frames
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "skip_frames": ("INT", {
                    "default": 1,  # process every N frames
                    "min": 1,
                    "max": 30,
                    "step": 1
                }),
                "show_body": ("BOOLEAN", {"default": True}),
                "show_hand": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "multiline": False}),  # 선택적으로 저장 경로 지정
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("openpose_video",)
    FUNCTION = "extract_openpose"
    CATEGORY = CATEGORY
    
    def __init__(self):
        self.openpose_detector = None
        self.mp_pose = None
        self.mp_hands = None
        self.mp_face = None
    
    def initialize_controlnet_openpose(self):
        """ControlNet OpenPose 초기화"""
        if not OPENPOSE_AVAILABLE:
            raise RuntimeError("controlnet_aux is not installed")
        
        if self.openpose_detector is None:
            print("[Mari - Video OpenPose] 🔧 Initializing ControlNet OpenPose detector...")
            self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        
        return self.openpose_detector
    
    def initialize_mediapipe(self, show_body, show_hand, show_face):
        """MediaPipe 초기화"""
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe is not installed")
        
        print("[Mari - Video OpenPose] 🔧 Initializing MediaPipe...")
        
        if show_body and self.mp_pose is None:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        if show_hand and self.mp_hands is None:
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        if show_face and self.mp_face is None:
            self.mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def draw_mediapipe_pose(self, image, results_pose, results_hands, results_face, 
                           show_body, show_hand, show_face):
        """MediaPipe 결과를 이미지에 그리기"""
        h, w, _ = image.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Body pose
        if show_body and results_pose and results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                canvas,
                results_pose.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Hands
        if show_hand and results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Face
        if show_face and results_face and results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    canvas,
                    face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        
        return canvas
    
    def extract_openpose(self, video, detection_method, fps, 
                        max_frames, skip_frames, show_body, show_hand, show_face,
                        output_path=""):
        """동영상에서 OpenPose 추출"""
        
        # VIDEO 타입 처리 - 다양한 형태 지원
        video_path = None
        
        print(f"[Mari - Video OpenPose] 🔍 DEBUG - Video type: {type(video)}")
        print(f"[Mari - Video OpenPose] 🔍 DEBUG - Video repr: {repr(video)}")
        
        # ComfyUI VideoFromFile 객체 또는 기타 객체
        if hasattr(video, '__dict__'):
            attrs = vars(video)
            print(f"[Mari - Video OpenPose] 🔍 DEBUG - Video attributes: {attrs}")
            
            # 모든 속성을 순회하며 경로처럼 보이는 것 찾기
            for key, value in attrs.items():
                if value and isinstance(value, str) and (
                    value.endswith('.mp4') or 
                    value.endswith('.avi') or 
                    value.endswith('.mov') or 
                    value.endswith('.mkv') or
                    value.endswith('.webm') or
                    '/' in value or 
                    '\\' in value
                ):
                    video_path = value
                    print(f"[Mari - Video OpenPose] ✅ Found path in attribute '{key}': {value}")
                    break
        
        # 딕셔너리 형태
        if video_path is None and isinstance(video, dict):
            print(f"[Mari - Video OpenPose] 🔍 DEBUG - Dict keys: {video.keys()}")
            video_path = video.get('video', video.get('path', video.get('file', video.get('filename', None))))
        
        # 문자열 형태
        if video_path is None and isinstance(video, str):
            video_path = video
        
        # 리스트/튜플 형태 (첫 번째 요소 확인)
        if video_path is None and isinstance(video, (list, tuple)) and len(video) > 0:
            print(f"[Mari - Video OpenPose] 🔍 DEBUG - List/Tuple, checking first element")
            return self.extract_openpose(video[0], detection_method, fps, max_frames, 
                                        skip_frames, show_body, show_hand, show_face, output_path)
        
        # 여전히 None이면 상세 디버그 정보 출력
        if video_path is None:
            print(f"[Mari - Video OpenPose] ❌ ERROR - Cannot find video path")
            print(f"[Mari - Video OpenPose] 🔍 DEBUG - Type: {type(video)}")
            print(f"[Mari - Video OpenPose] 🔍 DEBUG - Dir: {dir(video)}")
            if hasattr(video, '__dict__'):
                print(f"[Mari - Video OpenPose] 🔍 DEBUG - Dict: {vars(video)}")
            raise ValueError(
                f"Cannot extract video path from input. "
                f"Type: {type(video)}, "
                f"Available methods: {[m for m in dir(video) if not m.startswith('_')]}"
            )
        
        print(f"[Mari - Video OpenPose] 📂 Extracted video path: {video_path}")
        
        # 출력 경로 설정 (지정 안 하면 임시 파일)
        if not output_path or output_path.strip() == "":
            import tempfile
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time())
            output_path = os.path.join(temp_dir, f"mari_openpose_{timestamp}.mp4")
            print(f"[Mari - Video OpenPose] 📁 No output path specified, using temp: {output_path}")
        
        print("=" * 60)
        print("[Mari - Video OpenPose] 🎬 Starting video OpenPose extraction...")
        print(f"  📂 Input: {video_path}")
        print(f"  💾 Output: {output_path}")
        print(f"  🎯 Method: {detection_method}")
        print("=" * 60)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # 동영상 열기
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # 동영상 정보
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[Mari - Video OpenPose] 📊 Video info:")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  Original FPS: {original_fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Output FPS: {fps}")
        print(f"  Skip frames: {skip_frames} (process every {skip_frames} frame(s))")
        
        # 출력 동영상 설정
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_path}")
        
        # 감지기 초기화
        if detection_method == "controlnet_openpose":
            detector = self.initialize_controlnet_openpose()
        elif detection_method == "mediapipe":
            self.initialize_mediapipe(show_body, show_hand, show_face)
        else:
            raise ValueError("No detection method available. Please install controlnet-aux or mediapipe.")
        
        # 프레임 처리
        processed_frames = 0
        skipped_frames = 0
        frame_idx = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # max_frames 체크
            if max_frames > 0 and processed_frames >= max_frames:
                print(f"[Mari - Video OpenPose] ⏹️ Reached max_frames limit: {max_frames}")
                break
            
            # skip_frames 적용
            if frame_idx % skip_frames != 0:
                skipped_frames += 1
                frame_idx += 1
                continue
            
            # RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # OpenPose 추출
            if detection_method == "controlnet_openpose":
                # ControlNet OpenPose
                pil_image = Image.fromarray(frame_rgb)
                pose_image = detector(pil_image, hand_and_face=show_hand or show_face)
                pose_array = np.array(pose_image)
                pose_bgr = cv2.cvtColor(pose_array, cv2.COLOR_RGB2BGR)
            
            elif detection_method == "mediapipe":
                # MediaPipe
                results_pose = None
                results_hands = None
                results_face = None
                
                if show_body:
                    results_pose = self.mp_pose.process(frame_rgb)
                if show_hand:
                    results_hands = self.mp_hands.process(frame_rgb)
                if show_face:
                    results_face = self.mp_face.process(frame_rgb)
                
                pose_bgr = self.draw_mediapipe_pose(
                    frame, results_pose, results_hands, results_face,
                    show_body, show_hand, show_face
                )
            
            # 프레임 저장
            out.write(pose_bgr)
            
            processed_frames += 1
            frame_idx += 1
            
            # 진행 상황 출력
            if processed_frames % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = processed_frames / elapsed if elapsed > 0 else 0
                print(f"[Mari - Video OpenPose] 📹 Processed: {processed_frames} frames ({fps_current:.1f} FPS)")
        
        # 리소스 정리
        cap.release()
        out.release()
        
        end_time = time.time()
        total_elapsed = end_time - start_time
        total_min = int(total_elapsed // 60)
        total_sec = int(total_elapsed % 60)
        avg_fps = processed_frames / total_elapsed if total_elapsed > 0 else 0
        
        print("=" * 60)
        print("[Mari - Video OpenPose] ✅ 작업 완료!")
        print(f"  🎬 처리된 프레임: {processed_frames}개")
        print(f"  ⏭️ 건너뛴 프레임: {skipped_frames}개")
        print(f"  🕒 총 소요 시간: {total_min}m {total_sec}s")
        print(f"  ⚡ 평균 처리 속도: {avg_fps:.2f} FPS")
        print(f"  💾 저장 경로: {output_path}")
        print("=" * 60)
        
        # VideoWrapper로 감싸서 반환
        return (VideoWrapper(output_path),)


class MariVideoFrameExtractor:
    """
    동영상에서 프레임을 추출하여 이미지 시퀀스로 저장하는 노드
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),  # VIDEO input 노드로 받기
                "skip_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "max_frames": ("INT", {
                    "default": 0,  # 0 = all
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
            },
            "optional": {
                "output_folder": ("STRING", {"default": "", "multiline": False}),  # 선택적으로 저장
                "output_format": (["png", "jpg", "webp"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "extract_frames"
    CATEGORY = CATEGORY
    
    def extract_frames(self, video, skip_frames, max_frames, output_folder="", output_format="png"):
        """동영상에서 프레임 추출"""
        
        # VIDEO 타입 처리 - 다양한 형태 지원
        video_path = None
        
        print(f"[Mari - Frame Extractor] 🔍 DEBUG - Video type: {type(video)}")
        
        # ComfyUI VideoFromFile 객체 또는 기타 객체
        if hasattr(video, '__dict__'):
            attrs = vars(video)
            print(f"[Mari - Frame Extractor] 🔍 DEBUG - Video attributes: {attrs}")
            
            # 모든 속성을 순회하며 경로처럼 보이는 것 찾기
            for key, value in attrs.items():
                if value and isinstance(value, str) and (
                    value.endswith('.mp4') or 
                    value.endswith('.avi') or 
                    value.endswith('.mov') or 
                    value.endswith('.mkv') or
                    value.endswith('.webm') or
                    '/' in value or 
                    '\\' in value
                ):
                    video_path = value
                    print(f"[Mari - Frame Extractor] ✅ Found path in attribute '{key}': {value}")
                    break
        
        # 딕셔너리 형태
        if video_path is None and isinstance(video, dict):
            video_path = video.get('video', video.get('path', video.get('file', video.get('filename', None))))
        
        # 문자열 형태
        if video_path is None and isinstance(video, str):
            video_path = video
        
        # 리스트/튜플 형태
        if video_path is None and isinstance(video, (list, tuple)) and len(video) > 0:
            return self.extract_frames(video[0], skip_frames, max_frames, output_folder, output_format)
        
        # 여전히 None이면 에러
        if video_path is None:
            raise ValueError(
                f"Cannot extract video path from input. "
                f"Type: {type(video)}, "
                f"Dir: {dir(video)}"
            )
        
        print(f"[Mari - Frame Extractor] 📂 Extracted video path: {video_path}")
        print(f"[Mari - Frame Extractor] 🎬 Extracting frames from: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # 선택적으로 폴더에 저장
        save_to_disk = output_folder and output_folder.strip() != ""
        if save_to_disk:
            os.makedirs(output_folder, exist_ok=True)
            print(f"[Mari - Frame Extractor] 💾 Saving frames to: {output_folder}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[Mari - Frame Extractor] 📊 Total frames: {total_frames}, FPS: {fps:.2f}")
        
        frame_idx = 0
        saved_count = 0
        frames_list = []
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if max_frames > 0 and saved_count >= max_frames:
                break
            
            if frame_idx % skip_frames == 0:
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # numpy to tensor (ComfyUI IMAGE format)
                frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)[None, ...]
                frames_list.append(frame_tensor)
                
                # 선택적으로 디스크에 저장
                if save_to_disk:
                    output_name = f"frame_{saved_count:06d}.{output_format}"
                    output_path = os.path.join(output_folder, output_name)
                    
                    if output_format == "jpg":
                        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    elif output_format == "webp":
                        cv2.imwrite(output_path, frame, [cv2.IMWRITE_WEBP_QUALITY, 95])
                    else:
                        cv2.imwrite(output_path, frame)
                
                saved_count += 1
                
                if saved_count % 100 == 0:
                    print(f"[Mari - Frame Extractor] 📹 Extracted: {saved_count} frames")
            
            frame_idx += 1
        
        cap.release()
        
        if save_to_disk:
            print(f"[Mari - Frame Extractor] ✅ Complete! Saved {saved_count} frames to {output_folder}")
        else:
            print(f"[Mari - Frame Extractor] ✅ Complete! Extracted {saved_count} frames")
        
        return (frames_list,)


# ComfyUI 노드 등록
NODE_CLASS_MAPPINGS = {
    "Mari - Video OpenPose Extractor": MariVideoOpenPoseExtractor,
    "Mari - Video Frame Extractor": MariVideoFrameExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mari - Video OpenPose Extractor": "Mari - Video OpenPose Extractor",
    "Mari - Video Frame Extractor": "Mari - Video Frame Extractor",
}
