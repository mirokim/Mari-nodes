# Mari Nodes for ComfyUI

## ✨ 포함된 노드

### 1. Mari Nodes - Color Toolkit
이미지 색감을 조절하거나 단색 이미지를 생성합니다.

**기능**
- 밝기 (Brightness)
- 대비 (Contrast)
- 채도 (Saturation)
- 감마 보정 (Gamma)
- 노출 (Exposure, EV 단위)
- 색상 회전 (Hue Shift)
- HEX 단색 이미지 생성 (`#RRGGBB` 코드)

**출력**
- 조정된 이미지 (IMAGE)

---

### 2. Mari Nodes - Image Resize
이미지를 다양한 모드와 프리셋으로 리사이즈합니다.

**기능**
- 모드: `by_size`, `by_scale`
- 비율 옵션: `stretch`, `fit`, `fill`, `longer_side`, `shorter_side`
- 해상도 프리셋: 512x512, 768x768, 1920x1080 등 (기본값: Custom)
- 정렬: 가운데, 상/하/좌/우 등
- 패딩 색상: HEX 코드로 지정
- Even size 강제 (홀수를 짝수로 맞춤)
- Round to multiple (지정 배수로 반올림)

**출력**
- 리사이즈된 이미지 (IMAGE)

---

### 3. Mari Nodes - Load Combo
체크포인트와 LoRA, VAE를 한 번에 불러오는 통합 노드입니다.

**기능**
- 체크포인트 로드 (드롭다운 + override 수동 입력 지원)
- LoRA 최대 6개 적용 (on/off + strength 조절)
- VAE 교체
- CLIP skip
- 모델 정보 자동 판별
  - 패밀리: **SD1.x / SD2.x / SDXL**
  - parameterization (eps/v), CLIP 타입(OpenCLIP 등)
  - 적용된 LoRA 목록 JSON 출력

**출력**
1. MODEL  
2. CLIP  
3. VAE  
4. STRING (family: "SD1.x", "SD2.x", "SDXL")  
5. STRING (info_json: variant, parameterization, clip_type, ckpt/vae/clip_skip, LoRA 목록)

---

