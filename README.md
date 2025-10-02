📦 1. Mari Nodes - Color Toolkit

이미지 색감을 조절하거나 단색 이미지를 생성할 수 있습니다.

기능

밝기 (Brightness): 이미지 전체의 밝기를 조절

대비 (Contrast): 픽셀 값의 대비를 강화/약화

채도 (Saturation): 색상의 선명도를 조절

감마 (Gamma): 감마 보정

노출 (Exposure): EV 값 기반 노출 조정 (1.0 = x2, -1.0 = x0.5)

색상 회전 (Hue Shift): 색상을 지정 각도만큼 회전

HEX 단색 이미지 생성: #RRGGBB 코드로 단색 캔버스 생성

출력

조정된 이미지 (IMAGE)

📦 2. Mari Nodes - Image Resize

이미지를 다양한 방식으로 리사이즈할 수 있습니다.

기능

리사이즈 모드

by_size: 지정된 크기로 리사이즈

by_scale: 배율로 리사이즈

비율 유지 옵션

stretch: 단순 비율 무시

fit: 비율 유지 + 패딩

fill: 비율 유지 + 크롭

longer_side: 긴 변 기준

shorter_side: 짧은 변 기준

해상도 프리셋: 512x512, 768x768, 1920x1080 등 자주 쓰이는 해상도 제공 (기본값: Custom)

정렬(align): 상하좌우/가운데 정렬

패딩 색상(pad_color): HEX 코드로 지정

even size 강제: 홀수 크기를 자동으로 짝수화

round to multiple: 해상도를 지정 배수 단위로 반올림

출력

리사이즈된 이미지 (IMAGE)

📦 3. Mari Nodes - Load Combo

체크포인트와 LoRA, VAE를 한 번에 불러오는 통합 노드입니다.

기능

체크포인트 로드

드롭다운 또는 수동 입력(override) 지원

LoRA 최대 6개 적용

각각 on/off, strength (model/clip) 조절 가능

VAE 교체: 기본 또는 지정 VAE 로드

CLIP skip: CLIP 레이어 스킵

모델 정보 판별

패밀리: SD1.x / SD2.x / SDXL

파라미터화 방식(eps/v-pred), CLIP 타입(OpenCLIP 등) 자동 표시

적용된 LoRA 목록 JSON 출력

출력

MODEL

CLIP

VAE

STRING (모델 패밀리: "SD1.x", "SD2.x", "SDXL")

STRING (확장 정보 JSON: variant, parameterization, clip_type, ckpt/vae/clip_skip, LoRA 적용 목록)

설치 방법

mari_nodes 폴더를 ComfyUI/custom_nodes/에 복사

ComfyUI 재시작

노드 검색창에서 아래 이름으로 사용 가능

Mari Nodes - Color Toolkit

Mari Nodes - Image Resize

Mari Nodes - Load Combo
