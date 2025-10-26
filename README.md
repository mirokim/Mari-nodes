# 📦 Mari Nodes — ComfyUI 커스텀 노드 모음

> **Mari의 개인용 툴킷**으로, 색보정, 이미지 리사이즈, 빈 잠재공간 생성,  
> 그리고 폴더 단위의 배치 이미지 처리를 간단하게 만들어줍니다.  
> 반복 작업을 줄이고 ComfyUI 워크플로우를 간결하게 만드는 데 초점을 두었습니다.

---

## 🧰 제공 노드

### 1. 🖼 Mari – Color Toolkit
- 이미지의 밝기, 대비, 채도, 감마, Hue를 빠르게 조절하고 원본과 자연스럽게 블렌드.
- HSV 변환 기반의 안전한 색조 변경.
- 파이프라인 어디에나 끼워넣기 쉬운 단일 노드.

---

### 2. 🪄 Mari – Image Resize
- **scale 모드**: 전체 비율 조절  
- **custom 모드**: 지정한 해상도로 리사이즈  
- 보간 방식 지원: bilinear, bicubic, nearest, area  
- 마스크 자동 생성으로 후단 호환성 보장.

---

### 3. 🧱 Mari – Empty Latent+
- 빈 Latent 이미지를 프리셋 또는 퍼센트 스케일로 생성.  
- 64 배수 자동 스냅, 4채널 고정.  
- 배치 크기 지정 지원으로 후속 작업 준비 용이.

---

### 4. 🧠 Mari – Batch Img2Img  ✅ *(New)*
- 폴더 내 이미지를 한 번에 불러와 i2i 변환을 자동 반복 실행.  
- 여러 denoise 값, 다양한 seed 전략 지원.  
- 원본 크기 출력 또는 배치 출력 선택 가능.

**핵심 옵션**
| 이름 | 설명 |
|------|------|
| `input_folder` / `output_folder` | 폴더 경로 |
| `denoise_values` | 쉼표로 구분된 값 (예: `0.5,0.6,0.7`) |
| `seed_mode` | fixed / per_denoise / per_image / per_image_and_denoise / random |
| `mode` | original / batch |
| `padding_color` | black / white / #RRGGBB |
| `sampler_name`, `scheduler` | 샘플링 설정 |
| `positive`, `negative` | 외부 CLIP Text Encode 입력 |

---

### 5. 🪄 Mari – Folder Image Scale ✅ *(New)*
- 폴더 내 모든 이미지를 **가장 긴 변 기준**으로 비율 유지하며 일괄 리사이즈.
- PNG, JPG, JPEG, WEBP 지원.
- 배치 처리 전 단계로 활용하기 좋음.

---

## 🪄 예시 워크플로우

```
[Mari - Folder Image Scale]
        └── output_folder ─▶ [Mari - Batch Img2Img].input_folder

[CLIP Text Encode] (positive) ─▶ [Mari - Batch Img2Img].positive
[CLIP Text Encode] (negative) ─▶ [Mari - Batch Img2Img].negative

[Load Checkpoint]
    ├─ model ──────────────▶ [Mari - Batch Img2Img]
    └─ vae   ──────────────▶ [Mari - Batch Img2Img]
```

---

## ⚠️ 주의사항
- `input_folder`에 이미지 외 파일이 있으면 오류가 발생할 수 있습니다.  
- denoise 개수가 많아질수록 처리 시간과 VRAM 사용량이 증가합니다.  
- batch 모드에서 첫 이미지 크기가 기준이 되므로, 결과물 크기가 통일됩니다.  
- seed_mode에 따라 결과가 달라질 수 있습니다.

---

## 📝 설치
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mirokim/Mari-nodes
```

또는 `mari_nodes` 폴더에 개별 `.py` 파일을 복사해도 됩니다.

---

## 🧑‍💻 제작자
- **Mari**
- 📍 [https://github.com/mirokim](https://github.com/mirokim)
