### 1) Mari – Color Toolkit
이미지의 **밝기/대비/채도/감마/색상(Hue)** 를 빠르게 조절하고, 결과를 원본과 **블렌드**하여 자연스럽게 적용할 수 있습니다.  
- 입력: `IMAGE`, (옵션) `global_blend`
- 출력: `IMAGE`
- 카테고리: `Mari/Color`

**핵심 포인트**
- HSV 변환으로 Hue Shift를 안전하게 적용
- 0~1 클램핑으로 색상 값 안정화
- 파이프라인 어디에나 끼워 넣기 쉬운 단일 노드

---

### 2) Mari – Image Resize
두 가지 모드로 이미지를 리사이즈합니다.
- **scale 모드**: `scale`(단일 값) 또는 `scale_x/scale_y(+lock_aspect)`
- **custom 모드**: `target_width/target_height` 로 강제 크기 지정

보간 방식: `bilinear`, `bicubic`, `nearest`, `area(다운샘플 전용)`  
- 입력: `IMAGE`, `mode`, `method`, `scale`, `scale_x`, `scale_y`, `lock_aspect`, `target_width`, `target_height`, (옵션)`MASK`
- 출력: `IMAGE`, `MASK`, `STRING(info)`
- 카테고리: `Mari/Image`

**핵심 포인트**
- 마스크가 없을 경우 자동으로 **검은 마스크(0)** 를 생성해 후단 호환성 보장
- `area` 모드는 업샘플 시 자동으로 bilinear로 대체하여 아티팩트 완화
- Info 문자열로 리사이즈 내역(최종 크기)을 즉시 확인

---

### 3) Mari – Empty Latent+
해상도 **프리셋**과 **퍼센트 스케일**을 지원하는 **빈 LATENT** 생성 노드입니다.  
사이즈는 내부적으로 **64 배수**로 스냅되어 파이프라인 호환성을 높였습니다.
- 입력: `preset`, `use_custom`, `width`, `height`, `scale_percent`, `batch_size`
- 출력: `LATENT`, `out_width(INT)`, `out_height(INT)`
- 카테고리: `Mari Nodes/Image`

**핵심 포인트**
- SD 규약에 맞춰 **항상 4채널** 잠재공간 생성
- FHD, 4K, 9:16 등 실전 프리셋 다수 내장
- 배치 크기 지정으로 멀티 샘플 준비 용이

---

### 4) Mari – Load Combo (CKPT + LoRA x8 + ver)
**체크포인트 1개 + LoRA 최대 8개**를 한 번에 로드하고, **마스터 스위치**와 **글로벌 LoRA 스케일**로 전체 강도를 일괄 제어합니다.  
체크포인트/LoRA의 **SD 버전(SDXL / SD 2.x / SD 1.x)** 을 이름으로 추정하여 표시합니다.
- 입력: `ckpt_name`, `enable_loras`, `global_lora_scale`, `lora[1..8]`, `lora[1..8]_strength_model/clip`, `guess_config`, `output_vae`, (옵션)`vae_name`
- 출력: `MODEL`, `CLIP`, `VAE`, `ckpt_version(STRING)`, `lora_versions(STRING-JSON)`
- 카테고리: `Mari/Loaders`

**핵심 포인트**
- `guess_config=True` 시 `comfy.sd.load_checkpoint_guess_config` 사용
- LoRA 개별 강도 × 글로벌 스케일 = 최종 강도
- LoRA 버전 리스트를 **JSON 문자열**로 출력 → 로그/디버깅/UI 표시에 활용

---
