# -*- coding: utf-8 -*-
"""
Mari Subject Position & Size Matcher Node for ComfyUI
레퍼런스 이미지의 피사체 위치와 크기를 타겟 이미지에 그대로 적용
"""

import torch
import numpy as np
from PIL import Image
import cv2

CATEGORY = "Mari Nodes"


class MariSubjectPositionSizeMatcher:
    """
    레퍼런스 이미지의 피사체 위치와 크기를 타겟 이미지에 그대로 적용하는 노드
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),  # 기준 (위치와 크기)
                "target_image": ("IMAGE",),      # 조정할 이미지
                "detection_method": (["bounding_box", "contour", "saliency"],),
                "background_color": (["white", "black", "reference", "target"],),
                "match_mode": (["position_and_size", "size_only", "position_only"],),
                "duplicate_removal": ("BOOLEAN", {"default": True}),
                "duplicate_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "display": "slider"
                }),
                "subject_selection": (["largest", "leftmost", "rightmost", "topmost", "bottommost", "centermost"],),
                "skip_if_similar": ("BOOLEAN", {"default": True}),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_image",)
    FUNCTION = "match_subject"
    CATEGORY = CATEGORY
    
    def tensor_to_pil(self, tensor):
        """ComfyUI 텐서를 PIL 이미지로 변환"""
        np_image = tensor[0].cpu().numpy()
        np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    def pil_to_tensor(self, pil_image):
        """PIL 이미지를 ComfyUI 텐서로 변환"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image)[None,]
        return tensor
    
    def detect_subject_bbox(self, image, method="bounding_box"):
        """이미지에서 피사체의 경계 박스를 감지"""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        if method == "bounding_box":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        elif method == "contour":
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        elif method == "saliency":
            try:
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliency_map) = saliency.computeSaliency(cv_image)
                saliency_map = (saliency_map * 255).astype("uint8")
                _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((5, 5), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except Exception as e:
                print(f"[Mari - Position Size Matcher] ⚠️ Saliency failed, fallback: {e}")
                return self.detect_subject_bbox(image, "bounding_box")
        
        if not contours:
            h, w = gray.shape
            print(f"[Mari - Position Size Matcher] ⚠️ No contours found, using full image")
            return (0, 0, w, h)
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)
    
    def detect_all_subjects(self, image, method="bounding_box", min_area=100):
        """이미지에서 모든 피사체의 경계 박스를 감지"""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        if method == "bounding_box":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        elif method == "contour":
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        elif method == "saliency":
            try:
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliency_map) = saliency.computeSaliency(cv_image)
                saliency_map = (saliency_map * 255).astype("uint8")
                _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((5, 5), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except Exception as e:
                print(f"[Mari - Position Size Matcher] ⚠️ Saliency failed, fallback: {e}")
                return self.detect_all_subjects(image, "bounding_box", min_area)
        
        # 모든 윤곽선을 bbox로 변환 (최소 면적 이상만)
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h, area))
        
        # 면적 기준으로 정렬 (큰 것부터)
        bboxes.sort(key=lambda b: b[4], reverse=True)
        
        return bboxes
    
    def are_duplicates(self, bbox1, bbox2, threshold=0.3):
        """두 bbox가 중복인지 판단 (크기와 위치 유사도 기반)"""
        x1, y1, w1, h1 = bbox1[:4]
        x2, y2, w2, h2 = bbox2[:4]
        
        # 크기 비율 비교
        size_ratio = min(w1*h1, w2*h2) / max(w1*h1, w2*h2)
        
        # 중심점 거리 계산
        center1_x, center1_y = x1 + w1/2, y1 + h1/2
        center2_x, center2_y = x2 + w2/2, y2 + h2/2
        
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        max_dimension = max(w1, h1, w2, h2)
        
        # 거리가 작고 크기가 비슷하면 중복으로 판단
        normalized_distance = distance / max_dimension if max_dimension > 0 else 1
        
        # threshold보다 크기가 비슷하고 거리가 가까우면 중복
        is_duplicate = size_ratio > (1 - threshold) and normalized_distance < threshold * 3
        
        return is_duplicate
    
    def remove_duplicates(self, bboxes, threshold=0.3):
        """중복된 bbox 제거"""
        if len(bboxes) <= 1:
            return bboxes
        
        unique_bboxes = []
        
        for bbox in bboxes:
            is_dup = False
            for unique_bbox in unique_bboxes:
                if self.are_duplicates(bbox, unique_bbox, threshold):
                    is_dup = True
                    print(f"[Mari - Position Size Matcher] 🔍 Duplicate detected and removed")
                    break
            
            if not is_dup:
                unique_bboxes.append(bbox)
        
        return unique_bboxes
    
    def select_subject(self, bboxes, selection_mode, image_size):
        """여러 bbox 중 하나를 선택"""
        if not bboxes:
            return None
        
        if len(bboxes) == 1:
            return bboxes[0]
        
        img_w, img_h = image_size
        
        if selection_mode == "largest":
            return max(bboxes, key=lambda b: b[4])  # area
        
        elif selection_mode == "leftmost":
            return min(bboxes, key=lambda b: b[0])  # x
        
        elif selection_mode == "rightmost":
            return max(bboxes, key=lambda b: b[0])  # x
        
        elif selection_mode == "topmost":
            return min(bboxes, key=lambda b: b[1])  # y
        
        elif selection_mode == "bottommost":
            return max(bboxes, key=lambda b: b[1])  # y
        
        elif selection_mode == "centermost":
            # 이미지 중앙에 가장 가까운 것
            center_x, center_y = img_w / 2, img_h / 2
            
            def distance_from_center(bbox):
                x, y, w, h = bbox[:4]
                bbox_center_x = x + w / 2
                bbox_center_y = y + h / 2
                return np.sqrt((bbox_center_x - center_x)**2 + (bbox_center_y - center_y)**2)
            
            return min(bboxes, key=distance_from_center)
        
        return bboxes[0]
    
    def extract_subject(self, image, bbox):
        """이미지에서 피사체 영역만 추출"""
        x, y, w, h = bbox
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        subject = cv_image[y:y+h, x:x+w]
        return Image.fromarray(cv2.cvtColor(subject, cv2.COLOR_BGR2RGB))
    
    def match_subject(self, reference_image, target_image, detection_method, 
                     background_color, match_mode, duplicate_removal, 
                     duplicate_threshold, subject_selection, skip_if_similar, 
                     similarity_threshold):
        """레퍼런스의 피사체 위치/크기를 타겟에 적용"""
        
        print(f"[Mari - Position Size Matcher] 🎯 Starting matching (mode: {match_mode})...")
        
        ref_pil = self.tensor_to_pil(reference_image)
        target_pil = self.tensor_to_pil(target_image)
        
        # 레퍼런스 피사체 감지 (단일)
        ref_bbox = self.detect_subject_bbox(ref_pil, detection_method)
        ref_x, ref_y, ref_w, ref_h = ref_bbox
        
        # 타겟 피사체 감지 (복수 가능)
        if duplicate_removal:
            print(f"[Mari - Position Size Matcher] 🔍 Detecting all subjects (duplicate removal ON)...")
            target_bboxes = self.detect_all_subjects(target_pil, detection_method)
            print(f"[Mari - Position Size Matcher] 📊 Found {len(target_bboxes)} subject(s)")
            
            # 중복 제거
            unique_bboxes = self.remove_duplicates(target_bboxes, duplicate_threshold)
            print(f"[Mari - Position Size Matcher] ✂️ After duplicate removal: {len(unique_bboxes)} subject(s)")
            
            # 피사체 선택
            selected_bbox = self.select_subject(unique_bboxes, subject_selection, target_pil.size)
            if selected_bbox is None:
                print(f"[Mari - Position Size Matcher] ⚠️ No subject found, using full image")
                target_bbox = (0, 0, target_pil.width, target_pil.height)
            else:
                target_bbox = selected_bbox[:4]  # (x, y, w, h) only
                print(f"[Mari - Position Size Matcher] 🎯 Selected subject by '{subject_selection}' criterion")
        else:
            # 기존 방식 (가장 큰 것만)
            target_bbox = self.detect_subject_bbox(target_pil, detection_method)
        
        target_x, target_y, target_w, target_h = target_bbox
        
        print(f"[Mari - Position Size Matcher] 📏 Reference bbox: x={ref_x}, y={ref_y}, w={ref_w}, h={ref_h}")
        print(f"[Mari - Position Size Matcher] 📏 Target bbox: x={target_x}, y={target_y}, w={target_w}, h={target_h}")
        
        # 크기 유사도 체크
        if skip_if_similar and match_mode in ["position_and_size", "size_only"]:
            # 크기 비율 계산 (면적 기준)
            ref_area = ref_w * ref_h
            target_area = target_w * target_h
            
            if ref_area > 0 and target_area > 0:
                size_ratio = min(ref_area, target_area) / max(ref_area, target_area)
                
                print(f"[Mari - Position Size Matcher] 📐 Size similarity: {size_ratio:.3f} (threshold: {similarity_threshold})")
                
                if size_ratio >= similarity_threshold:
                    print(f"[Mari - Position Size Matcher] ⏭️ SKIP! Sizes are already similar (ratio: {size_ratio:.3f})")
                    print(f"[Mari - Position Size Matcher] ✅ Returning original target image")
                    return (target_image,)
        
        # 타겟 피사체 추출
        target_subject = self.extract_subject(target_pil, target_bbox)
        
        # 크기 조정 (비율 유지)
        if match_mode in ["position_and_size", "size_only"]:
            # 레퍼런스 크기에 맞춤 (비율 유지)
            # 원본 피사체의 비율
            original_aspect = target_w / target_h if target_h > 0 else 1
            target_aspect = ref_w / ref_h if ref_h > 0 else 1
            
            # 비율 유지하면서 레퍼런스 bbox에 맞춤
            if original_aspect > target_aspect:
                # 너비 기준
                new_w = ref_w
                new_h = int(ref_w / original_aspect)
            else:
                # 높이 기준
                new_h = ref_h
                new_w = int(ref_h * original_aspect)
            
            target_subject = target_subject.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"[Mari - Position Size Matcher] 📐 Resized subject to {new_w}x{new_h} (aspect ratio preserved)")
        else:
            new_w, new_h = target_subject.size
        
        # 배경색 결정
        if background_color == "white":
            bg_color = (255, 255, 255)
        elif background_color == "black":
            bg_color = (0, 0, 0)
        elif background_color == "reference":
            ref_array = np.array(ref_pil)
            edges = np.concatenate([
                ref_array[0, :, :],
                ref_array[-1, :, :],
                ref_array[:, 0, :],
                ref_array[:, -1, :]
            ])
            bg_color = tuple(edges.mean(axis=0).astype(int).tolist())
            print(f"[Mari - Position Size Matcher] 🎨 Using reference background: RGB{bg_color}")
        elif background_color == "target":
            target_array = np.array(target_pil)
            edges = np.concatenate([
                target_array[0, :, :],
                target_array[-1, :, :],
                target_array[:, 0, :],
                target_array[:, -1, :]
            ])
            bg_color = tuple(edges.mean(axis=0).astype(int).tolist())
            print(f"[Mari - Position Size Matcher] 🎨 Using target background: RGB{bg_color}")
        else:
            bg_color = (255, 255, 255)
        
        # 최종 이미지 생성 (타겟 원본 크기 유지!)
        result = Image.new('RGB', target_pil.size, bg_color)
        
        # 위치 결정
        if match_mode in ["position_and_size", "position_only"]:
            # 레퍼런스 위치에 배치 (중앙 정렬)
            paste_x = ref_x + (ref_w - new_w) // 2
            paste_y = ref_y + (ref_h - new_h) // 2
            print(f"[Mari - Position Size Matcher] 📍 Placing at reference position: ({paste_x}, {paste_y})")
        else:
            # 중앙 배치
            paste_x = (target_pil.width - new_w) // 2
            paste_y = (target_pil.height - new_h) // 2
            print(f"[Mari - Position Size Matcher] 📍 Placing at center: ({paste_x}, {paste_y})")
        
        # 피사체 붙여넣기
        result.paste(target_subject, (paste_x, paste_y))
        
        result_tensor = self.pil_to_tensor(result)
        
        print(f"[Mari - Position Size Matcher] ✅ Complete! Output size: {result.size}")
        
        return (result_tensor,)


class MariSubjectAligner:
    """
    두 이미지의 피사체를 같은 위치와 크기로 정렬 (비교용)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "detection_method": (["bounding_box", "contour", "saliency"],),
                "output_mode": (["side_by_side", "overlay", "separate"],),
                "background_color": (["white", "black", "transparent"],),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned_image1", "aligned_image2")
    FUNCTION = "align_subjects"
    CATEGORY = CATEGORY
    
    def tensor_to_pil(self, tensor):
        np_image = tensor[0].cpu().numpy()
        np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    def pil_to_tensor(self, pil_image):
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image)[None,]
        return tensor
    
    def detect_subject_bbox(self, image, method="bounding_box"):
        """이미지에서 피사체의 경계 박스를 감지"""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        if method == "bounding_box":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif method == "contour":
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            try:
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliency_map) = saliency.computeSaliency(cv_image)
                saliency_map = (saliency_map * 255).astype("uint8")
                _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except:
                return self.detect_subject_bbox(image, "bounding_box")
        
        if not contours:
            h, w = gray.shape
            return (0, 0, w, h)
        
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)
    
    def align_subjects(self, image1, image2, detection_method, output_mode, background_color):
        """두 이미지의 피사체를 같은 위치/크기로 정렬"""
        
        print(f"[Mari - Subject Aligner] 🎯 Aligning subjects...")
        
        img1_pil = self.tensor_to_pil(image1)
        img2_pil = self.tensor_to_pil(image2)
        
        # 피사체 감지
        bbox1 = self.detect_subject_bbox(img1_pil, detection_method)
        bbox2 = self.detect_subject_bbox(img2_pil, detection_method)
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 통합 크기 계산 (더 큰 쪽)
        max_w = max(w1, w2)
        max_h = max(h1, h2)
        
        # 캔버스 크기 (여유 공간 포함)
        canvas_w = max(img1_pil.width, img2_pil.width)
        canvas_h = max(img1_pil.height, img2_pil.height)
        
        # 배경색
        if background_color == "white":
            bg = (255, 255, 255)
        elif background_color == "black":
            bg = (0, 0, 0)
        else:
            bg = (128, 128, 128)
        
        # 중앙 위치
        center_x = (canvas_w - max_w) // 2
        center_y = (canvas_h - max_h) // 2
        
        # 이미지1 처리
        subject1 = img1_pil.crop((x1, y1, x1+w1, y1+h1))
        subject1_resized = subject1.resize((max_w, max_h), Image.Resampling.LANCZOS)
        result1 = Image.new('RGB', (canvas_w, canvas_h), bg)
        result1.paste(subject1_resized, (center_x, center_y))
        
        # 이미지2 처리
        subject2 = img2_pil.crop((x2, y2, x2+w2, y2+h2))
        subject2_resized = subject2.resize((max_w, max_h), Image.Resampling.LANCZOS)
        result2 = Image.new('RGB', (canvas_w, canvas_h), bg)
        result2.paste(subject2_resized, (center_x, center_y))
        
        print(f"[Mari - Subject Aligner] ✅ Aligned to {max_w}x{max_h} at center")
        
        return (self.pil_to_tensor(result1), self.pil_to_tensor(result2))


# ComfyUI 노드 등록
NODE_CLASS_MAPPINGS = {
    "Mari - Subject Position Size Matcher": MariSubjectPositionSizeMatcher,
    "Mari - Subject Aligner": MariSubjectAligner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mari - Subject Position Size Matcher": "Mari - Subject Position Size Matcher",
    "Mari - Subject Aligner": "Mari - Subject Aligner",
}
