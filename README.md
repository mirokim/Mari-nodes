# Mari Nodes — ComfyUI 커스텀 노드 모음

> **Mari의 개인용 툴킷**으로, 색보정·리사이즈·배치 처리·영상 분석 등 반복 작업을 줄이고
> ComfyUI 워크플로우를 간결하게 만드는 데 초점을 두었습니다.

---

## 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mirokim/Mari-nodes
```

또는 `Mari-nodes` 폴더에 개별 `.py` 파일을 복사해도 됩니다.

---

## 제공 노드 목록

| 카테고리 | 노드 이름 | 간단 설명 |
|----------|-----------|-----------|
| Color | Mari – Color Toolkit | 밝기·대비·채도·감마·색조 조절 |
| Image | Mari – Image Resize | 스케일/커스텀 리사이즈 (마스크 포함) |
| Image | Mari – Smart Resize | 조건부 자동 리사이즈 |
| Image | Mari – Empty Latent+ | 프리셋/퍼센트 빈 Latent 생성 |
| Image | Mari – Sequential Image Loader | 폴더 이미지 순차 로딩 |
| Image | Mari – Batch Img2Img | 폴더 단위 배치 Img2Img |
| Image | Mari – Folder Image Scale | 폴더 이미지 일괄 리사이즈 |
| Image | Mari – Subject Position Size Matcher | 피사체 위치·크기 매칭 |
| Image | Mari – Subject Aligner | 두 이미지 피사체 정렬 |
| Conditioning | Mari – CLIP Text Encode (Auto Save) | CLIP 인코딩 + 프롬프트 자동 저장 |
| Loaders | Mari – Load Combo | 체크포인트 + LoRA x8 한 번에 로드 |
| Video | Mari – Video Frame Extractor | 영상에서 프레임 추출 (IMAGE 배치 출력) |
| Video | Mari – Video OpenPose Extractor | 영상 OpenPose 추출 → 영상 출력 |
| Comic | Mari – Comic Grid Splitter | 그리드 이미지를 컷별로 분리 저장 |
| Utility | Mari – Delay | 지정 시간 대기 (패스스루 노드) |

---

## 노드 상세 설명

### Mari – Color Toolkit
이미지의 색 속성을 파이프라인 어디서나 빠르게 조절합니다.

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `brightness` | 밝기 배율 | 1.0 |
| `contrast` | 대비 배율 | 1.0 |
| `saturation` | 채도 배율 | 1.0 |
| `gamma` | 감마 보정 | 1.0 |
| `hue_shift` | 색조 이동 (도, -180 ~ 180) | 0.0 |
| `global_blend` | 원본과 블렌드 비율 (0=원본, 1=완전 적용) | 1.0 |

---

### Mari – Image Resize
두 가지 모드로 이미지를 리사이즈합니다. 마스크가 연결되면 함께 리사이즈됩니다.

| 파라미터 | 설명 |
|----------|------|
| `mode` | `scale` (배율) / `custom` (지정 해상도) |
| `scale` | 균일 배율 (scale 모드) |
| `scale_x` / `scale_y` | 축별 배율, `lock_aspect` 로 가로세로 연동 |
| `target_width` / `target_height` | 목표 해상도 (custom 모드) |
| `method` | bilinear / bicubic / nearest / area |

---

### Mari – Smart Resize
이미지가 지정 크기(`trigger_size`) 이상일 때만 자동으로 리사이즈합니다.

| 파라미터 | 설명 |
|----------|------|
| `trigger_size` | 이 크기 이상일 때만 작동 |
| `resize_mode` | `based_on_longest` / `based_on_width` / `based_on_height` / `by_percentage` |
| `target_pixels` | 목표 픽셀값 (픽셀 모드) |
| `target_percent` | 목표 비율 % (퍼센트 모드) |
| `method` | lanczos / nearest-exact / bilinear / area / bicubic |

---

### Mari – Empty Latent+
빈 Latent 이미지를 프리셋 또는 커스텀 크기로 생성합니다.

- 해상도 프리셋 드롭다운 제공 (512~4K, 가로세로 다양)
- `use_custom`으로 원하는 크기 직접 입력
- `scale_percent`로 퍼센트 스케일 적용
- 내부적으로 64의 배수로 자동 스냅
- 배치 크기 지정 가능

출력: `LATENT`, `out_width`, `out_height`

---

### Mari – Sequential Image Loader
폴더 안의 이미지를 실행할 때마다 한 장씩 순차적으로 불러옵니다.

- `start_at_index`를 바꾸면 해당 번호로 점프
- Loop 구조: 마지막 이미지 다음엔 첫 번째로 돌아옴
- 출력: `image`, `filename`, `current_index`

---

### Mari – Batch Img2Img
폴더 내 이미지를 자동으로 순회하며 Img2Img 변환을 일괄 처리합니다.

| 파라미터 | 설명 |
|----------|------|
| `input_folder` / `output_folder` | 입력/출력 폴더 경로 |
| `denoise_values` | 쉼표 구분 값 (예: `0.5,0.6,0.7`) |
| `seed_mode` | fixed / per_denoise / per_image / per_image_and_denoise / random |
| `mode` | `original` (각자 원본 크기) / `batch` (첫 이미지 크기로 통일) |
| `padding_color` | black / white / #RRGGBB |
| `sampler_name` / `scheduler` | 샘플러·스케줄러 |
| `positive` / `negative` | CLIP Text Encode 입력 |

---

### Mari – Folder Image Scale
폴더 내 모든 이미지를 **가장 긴 변 기준**으로 비율 유지하며 일괄 리사이즈합니다.

- PNG, JPG, JPEG, WEBP 지원
- `target_size`: 긴 변의 목표 픽셀
- `interpolation`: lanczos / bilinear / bicubic / nearest
- 출력: 저장된 폴더 경로 (다음 노드로 연결 가능)

---

### Mari – Subject Position Size Matcher
레퍼런스 이미지의 피사체 위치·크기를 타겟 이미지에 그대로 적용합니다.

| 파라미터 | 설명 |
|----------|------|
| `detection_method` | bounding_box / contour / saliency |
| `match_mode` | position_and_size / size_only / position_only |
| `background_color` | white / black / reference / target |
| `duplicate_removal` | 중복 피사체 제거 여부 |
| `subject_selection` | largest / leftmost / rightmost / topmost / bottommost / centermost |
| `skip_if_similar` | 이미 비슷한 크기면 건너뜀 |

---

### Mari – Subject Aligner
두 이미지의 피사체를 같은 위치와 크기로 정렬합니다 (비교·합성용).

- 두 이미지에서 각각 피사체 감지
- 더 큰 피사체 기준으로 두 이미지 모두 정렬
- 출력: `aligned_image1`, `aligned_image2`

---

### Mari – CLIP Text Encode (Auto Save)
기본 CLIPTextEncode 기능에 프롬프트 자동 저장을 추가한 노드입니다.

- 텍스트가 바뀔 때마다 자동으로 파일에 저장 (동일 내용 중복 저장 방지)
- `save_format`: json (히스토리 누적) / txt (타임스탬프 기록)
- `save_dir` 미입력 시 `output/mari_prompt_logs/`에 저장
- `label`로 각 프롬프트에 메모 추가 가능
- 출력: `conditioning`, `saved_path`

---

### Mari – Load Combo
체크포인트와 최대 8개의 LoRA를 하나의 노드에서 한 번에 로드합니다.

| 파라미터 | 설명 |
|----------|------|
| `ckpt_name` | 체크포인트 선택 |
| `enable_loras` | LoRA 전체 활성/비활성 마스터 스위치 |
| `global_lora_scale` | 모든 LoRA 강도에 곱해지는 글로벌 배율 |
| `lora1` ~ `lora8` | LoRA 파일 선택 (None이면 미사용) |
| `lora{n}_strength_model` / `_clip` | 개별 LoRA 강도 |
| `guess_config` | 자동 설정 감지 여부 |
| `vae_name` | 외부 VAE 지정 (선택) |

출력: `model`, `clip`, `vae`, `ckpt_version`, `lora_versions` (JSON 문자열)

---

### Mari – Video Frame Extractor
영상 파일 또는 VHS/AnimateDiff 등의 IMAGE 배치에서 프레임을 추출합니다.

- `frame_interval`: N프레임마다 1장 추출
- `start_frame` / `max_frames`: 구간 지정
- `save_format`: png / jpg / webp
- `images` 입력 연결 시 → IMAGE 배치 처리
- `video_path_text` 직접 입력도 가능
- 출력: `images` (배치 텐서), `output_path`, `frame_count`

---

### Mari – Video OpenPose Extractor
영상에서 OpenPose 스켈레톤을 프레임별로 추출하여 새 영상으로 출력합니다.

**의존성 설치 필요:**
```bash
pip install controlnet-aux   # controlnet_openpose 방식
pip install mediapipe        # mediapipe 방식
```

| 파라미터 | 설명 |
|----------|------|
| `detection_method` | controlnet_openpose / mediapipe |
| `skip_frames` | N프레임마다 1장 처리 |
| `max_frames` | 최대 처리 프레임 수 (0=전체) |
| `show_body` / `show_hand` / `show_face` | 감지 부위 선택 |
| `output_path` | 결과 영상 저장 경로 (미입력 시 임시 파일) |

---

### Mari – Comic Grid Splitter
그리드 형태의 이미지(예: 2행 3열)를 개별 컷 이미지로 분리합니다.

| 파라미터 | 설명 |
|----------|------|
| `rows` / `columns` | 행/열 수 |
| `filename_prefix` | 저장 파일명 앞에 붙는 접두사 |
| `output_path` | 저장 경로 (미입력 시 ComfyUI output 폴더) |
| `save_to_disk` | 파일 저장 여부 |

- 파일명 형식: `{prefix}_{전체카운터:04d}_{컷번호:02d}.jpg`
- RGBA 이미지는 자동으로 흰색 배경으로 변환
- 출력: 분리된 컷 이미지 배치 텐서

---

### Mari – Delay
지정한 시간(초) 동안 실행을 일시 정지합니다. 모든 입력을 그대로 통과시킵니다.

- GPU 과열 방지, 순차 실행 제어 등에 활용
- 지원 타입: IMAGE, LATENT, MODEL, CLIP, VAE, CONDITIONING, MASK
- `delay_seconds`: 0 ~ 300초

---

## 예시 워크플로우

```
[Mari - Folder Image Scale]
        └── output_folder ──▶ [Mari - Batch Img2Img].input_folder

[CLIP Text Encode] (positive) ──▶ [Mari - Batch Img2Img]
[CLIP Text Encode] (negative) ──▶ [Mari - Batch Img2Img]

[Mari – Load Combo]
    ├── model ──▶ [Mari - Batch Img2Img]
    └── vae   ──▶ [Mari - Batch Img2Img]
```

---

## 주의사항

- `input_folder`에 이미지 외 파일이 있으면 오류가 발생할 수 있습니다.
- Video OpenPose 노드는 `controlnet-aux` 또는 `mediapipe`가 설치되어 있어야 합니다.
- denoise 개수가 많아질수록 처리 시간과 VRAM 사용량이 증가합니다.

---

## 제작자

- **Mari**
- [https://github.com/mirokim](https://github.com/mirokim)
