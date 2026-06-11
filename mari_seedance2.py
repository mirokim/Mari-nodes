# -*- coding: utf-8 -*-
"""
Mari Seedance 2.0 node for ComfyUI.

Thin HTTP client for ByteDance's Seedance 2.0 video generation API
(BytePlus ModelArk / Volcengine Ark). Submits an async generation task,
polls until it succeeds, downloads the mp4 into ComfyUI's output folder
and decodes it into an IMAGE frame batch.

Supported generation modes (decided by which inputs are connected):
    - text-to-video            : prompt only
    - image-to-video           : first_frame (+ optional last_frame)
    - multimodal reference     : reference_images batch, reference video/audio URLs

API contract (POST /api/v3/contents/generations/tasks, then GET .../tasks/{id}):
    images are sent inline as base64 data URLs; up to 9 images total,
    3 reference videos, 3 reference audios. Result video URL expires
    after 24h, so the mp4 is always downloaded immediately.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Optional

import numpy as np
import torch

try:
    import folder_paths
    _OUTPUT_DIR = folder_paths.get_output_directory()
except Exception:
    _OUTPUT_DIR = os.path.join(os.getcwd(), "output")

try:
    import requests
except ImportError as e:
    raise ImportError(
        "[Mari Seedance2] 'requests' is required. Install with `pip install requests`."
    ) from e

# Corporate networks often intercept TLS with a root CA that lives in the
# Windows cert store but not in certifi's bundle -> CERTIFICATE_VERIFY_FAILED.
# truststore makes Python validate against the OS store instead.
try:
    import truststore
    truststore.inject_into_ssl()
    print("[Mari Seedance2] using OS certificate store (truststore)")
except ImportError:
    pass

from PIL import Image


_ENDPOINTS = {
    "BytePlus (Global)": {
        "base": "https://ark.ap-southeast.bytepluses.com/api/v3",
        "models": {
            "seedance-2.0": "dreamina-seedance-2-0-260128",
            "seedance-2.0-fast": "dreamina-seedance-2-0-fast-260128",
        },
    },
    "Volcengine (China)": {
        "base": "https://ark.cn-beijing.volces.com/api/v3",
        "models": {
            "seedance-2.0": "doubao-seedance-2-0-260128",
            "seedance-2.0-fast": "doubao-seedance-2-0-fast-260128",
        },
    },
}

_POLL_INTERVAL = 8          # seconds between status checks
_MAX_WAIT = 60 * 30         # give up after 30 minutes
_TIMEOUT_HTTP = 60
_MAX_IMAGES = 9             # API limit: 9 images per request


def _interrupt_check():
    """Raise if the user pressed Cancel in ComfyUI (no-op outside ComfyUI)."""
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _progress_text(text: str, node_id) -> None:
    """Show live status text under the node (no-op on old ComfyUI / outside ComfyUI)."""
    if node_id is None:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_progress_text(text, node_id)
    except Exception:
        pass


def _tensor_to_data_url(frame: torch.Tensor) -> str:
    """Single frame (H,W,C float 0-1) -> base64 PNG data URL."""
    arr = np.clip(frame.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr[..., :3], mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _image_entry(frame: torch.Tensor, role: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": _tensor_to_data_url(frame)},
        "role": role,
    }


_MAX_VIDEO_BYTES = 50 * 1024 * 1024   # API limit per video
_MAX_AUDIO_BYTES = 15 * 1024 * 1024   # API limit per audio


def _video_to_data_url(video) -> str:
    """ComfyUI VIDEO input (VideoFromFile/VideoFromComponents or path str) -> data URL."""
    data: Optional[bytes] = None
    ext = ".mp4"

    if isinstance(video, str):                      # plain file path
        ext = os.path.splitext(video)[1].lower() or ".mp4"
        with open(video, "rb") as f:
            data = f.read()
    elif hasattr(video, "get_stream_source"):       # VideoFromFile
        src = video.get_stream_source()
        if isinstance(src, str):
            ext = os.path.splitext(src)[1].lower() or ".mp4"
            with open(src, "rb") as f:
                data = f.read()
        else:                                       # BytesIO-like
            pos = src.tell()
            src.seek(0)
            data = src.read()
            src.seek(pos)
    if data is None and hasattr(video, "save_to"):  # VideoFromComponents etc.
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "mari_seedance2_ref.mp4")
        video.save_to(tmp)
        with open(tmp, "rb") as f:
            data = f.read()
        os.remove(tmp)
    if data is None:
        raise ValueError(f"[Mari Seedance2] unsupported VIDEO input type: {type(video)}")

    if len(data) > _MAX_VIDEO_BYTES:
        raise ValueError(f"[Mari Seedance2] reference video too large "
                         f"({len(data) // (1 << 20)}MB > 50MB API limit) — "
                         f"trim it or host it and use reference_video_url instead")
    mime = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
            ".avi": "video/x-msvideo", ".mkv": "video/x-matroska"}.get(ext, "video/mp4")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _audio_to_data_url(audio: dict) -> str:
    """ComfyUI AUDIO dict {waveform: (B,C,T), sample_rate: int} -> WAV data URL."""
    import wave
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    if wf.ndim == 3:
        wf = wf[0]                                  # (C, T)
    arr = wf.detach().cpu().numpy()
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")  # int16 little-endian
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(pcm.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.T.tobytes())              # interleave channels
    data = buf.getvalue()
    if len(data) > _MAX_AUDIO_BYTES:
        raise ValueError(f"[Mari Seedance2] reference audio too large "
                         f"({len(data) // (1 << 20)}MB > 15MB API limit) — "
                         f"trim it or host it and use reference_audio_url instead")
    return f"data:audio/wav;base64,{base64.b64encode(data).decode('ascii')}"


def _api_error(resp: requests.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return f"{err.get('code', resp.status_code)}: {err.get('message', resp.text[:300])}"
    except Exception:
        return f"HTTP {resp.status_code}: {resp.text[:300]}"


def _decode_video(path: str):
    """mp4 -> ((N,H,W,C) float tensor, fps). Tries av, then cv2, then imageio."""
    # 1) PyAV — bundled with recent ComfyUI
    try:
        import av
        frames = []
        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 24.0
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
        if frames:
            batch = np.stack(frames).astype(np.float32) / 255.0
            return torch.from_numpy(batch), fps
    except ImportError:
        pass

    # 2) OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            batch = np.stack(frames).astype(np.float32) / 255.0
            return torch.from_numpy(batch), float(fps)
    except ImportError:
        pass

    # 3) imageio
    try:
        import imageio.v3 as iio
        frames = iio.imread(path, plugin="pyav")
        meta = iio.immeta(path, plugin="pyav")
        batch = np.asarray(frames).astype(np.float32) / 255.0
        return torch.from_numpy(batch), float(meta.get("fps", 24.0))
    except ImportError:
        pass

    raise RuntimeError(
        "[Mari Seedance2] no video decoder available — install one of: av, opencv-python, imageio[pyav]"
    )


class MariSeedance2:
    """Generate video with ByteDance Seedance 2.0 via the Ark API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False,
                                       "tooltip": "Ark API key. Leave empty to use the ARK_API_KEY environment variable."}),
                "endpoint": (list(_ENDPOINTS.keys()),),
                "model": (["seedance-2.0", "seedance-2.0-fast"],),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "Reference connected inputs as [image 1], [image 2] ... in order."}),
                "resolution": (["480p", "720p", "1080p", "2K"], {"default": "1080p"}),
                "ratio": (["adaptive", "16:9", "9:16", "4:3", "3:4", "21:9", "1:1"], {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1,
                                     "tooltip": "Video length in seconds"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF,
                                 "control_after_generate": True,
                                 "tooltip": "-1 = random (server side)"}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "save_folder": ("STRING", {"default": "", "multiline": False,
                                           "tooltip": "Folder to save the mp4. Empty = ComfyUI output dir. "
                                                      "Relative paths are placed under the output dir."}),
                "ssl_verify": ("BOOLEAN", {"default": True,
                                           "tooltip": "Turn off only if you get CERTIFICATE_VERIFY_FAILED on a "
                                                      "corporate/proxy network and truststore doesn't help. "
                                                      "Disabling skips TLS certificate checks (insecure)."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "First frame for image-to-video"}),
                "last_frame": ("IMAGE", {"tooltip": "Last frame (requires first_frame)"}),
                "reference_images": ("IMAGE", {"tooltip": "Style/subject reference batch (up to 9 images total per request)"}),
                "reference_video": ("VIDEO", {"tooltip": "Reference video (motion/style), sent inline as base64 — max 50MB"}),
                "reference_audio": ("AUDIO", {"tooltip": "Reference audio (ambient/music), sent inline as WAV — max 15MB. "
                                              "Must accompany at least one image or video reference."}),
                "reference_video_url": ("STRING", {"default": "", "multiline": False,
                                                   "tooltip": "Public URL of a reference video (max 3, comma separated)"}),
                "reference_audio_url": ("STRING", {"default": "", "multiline": False,
                                                   "tooltip": "Public URL of a reference audio (max 3, comma separated)"}),
                "model_id_override": ("STRING", {"default": "", "multiline": False,
                                                 "tooltip": "Exact Ark model ID / inference endpoint ID. Overrides the model dropdown."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("frames", "video_path", "fps", "info", "tokens")
    FUNCTION = "generate"
    CATEGORY = "Mari/API"
    OUTPUT_NODE = True

    # ------------------------------------------------------------------
    def generate(self, api_key, endpoint, model, prompt, resolution, ratio,
                 duration, seed, generate_audio, watermark, save_folder="",
                 ssl_verify=True,
                 first_frame: Optional[torch.Tensor] = None,
                 last_frame: Optional[torch.Tensor] = None,
                 reference_images: Optional[torch.Tensor] = None,
                 reference_video=None,
                 reference_audio: Optional[dict] = None,
                 reference_video_url: str = "",
                 reference_audio_url: str = "",
                 model_id_override: str = "",
                 unique_id=None):

        if not ssl_verify:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            print("[Mari Seedance2] ⚠ TLS certificate verification DISABLED")

        key = api_key.strip() or os.environ.get("ARK_API_KEY", "").strip()
        if not key:
            raise ValueError("[Mari Seedance2] missing API key (node input or ARK_API_KEY env var)")

        ep = _ENDPOINTS[endpoint]
        model_id = model_id_override.strip() or ep["models"][model]
        base = ep["base"]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        content = self._build_content(prompt, first_frame, last_frame, reference_images,
                                      reference_video, reference_audio,
                                      reference_video_url, reference_audio_url)

        body = {
            "model": model_id,
            "content": content,
            "resolution": resolution,
            "ratio": ratio,
            "duration": int(duration),
            "generate_audio": bool(generate_audio),
            "watermark": bool(watermark),
        }
        if seed >= 0:
            body["seed"] = int(seed)

        # -- submit -----------------------------------------------------
        n_img = sum(1 for c in content if c["type"] == "image_url")
        print(f"[Mari Seedance2] submitting task: model={model_id} "
              f"{resolution} {ratio} {duration}s images={n_img}")
        _progress_text("submitting task...", unique_id)
        r = requests.post(f"{base}/contents/generations/tasks",
                          headers=headers, data=json.dumps(body),
                          timeout=_TIMEOUT_HTTP, verify=ssl_verify)
        if r.status_code != 200:
            raise RuntimeError(f"[Mari Seedance2] task creation failed — {_api_error(r)}")
        task_id = r.json()["id"]
        print(f"[Mari Seedance2] task created: {task_id}")

        # -- poll -------------------------------------------------------
        result = self._poll(base, headers, task_id, ssl_verify, unique_id)

        # -- download + decode -----------------------------------------
        video_url = result.get("content", {}).get("video_url")
        if not video_url:
            raise RuntimeError(f"[Mari Seedance2] task succeeded but no video_url in response: {result}")

        out_dir = self._resolve_save_dir(save_folder)
        out_path = os.path.join(out_dir, f"seedance2_{task_id}.mp4")
        with requests.get(video_url, stream=True, timeout=_TIMEOUT_HTTP * 5,
                          verify=ssl_verify) as dl:
            dl.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"[Mari Seedance2] video saved -> {out_path}")

        _progress_text("decoding frames...", unique_id)
        frames, fps = _decode_video(out_path)
        tokens = int(result.get("usage", {}).get("total_tokens", 0) or 0)
        # rough estimate at the standard T2V/I2V rate (~$6.4 per 1M tokens);
        # fast model and video-reference tasks are billed lower
        cost = tokens / 1_000_000 * 6.4
        info = (f"task={task_id} | {frames.shape[0]} frames @ {fps:.6g} fps | "
                f"tokens={tokens:,} (~${cost:.2f}) | saved: {out_path}")
        print(f"[Mari Seedance2] usage: {tokens:,} tokens (~${cost:.2f})")
        _progress_text(f"done — {frames.shape[0]} frames | "
                       f"{tokens:,} tokens (~${cost:.2f})", unique_id)

        # register the mp4 with the UI (history/output panel) — only possible
        # for files inside the ComfyUI output directory
        ui_videos = []
        rel = os.path.relpath(out_path, _OUTPUT_DIR)
        if not rel.startswith(".."):
            subfolder, fname = os.path.split(rel)
            ui_videos.append({"filename": fname, "subfolder": subfolder, "type": "output"})
        else:
            print("[Mari Seedance2] note: save_folder is outside the ComfyUI output dir, "
                  "so the video won't appear in the output/assets panel")

        return {"ui": {"images": ui_videos, "animated": (True,)},
                "result": (frames, out_path, fps, info, tokens)}

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_save_dir(save_folder: str) -> str:
        """Empty -> ComfyUI output dir; relative -> under output dir; absolute -> as-is."""
        folder = save_folder.strip().strip('"')
        if not folder:
            out_dir = _OUTPUT_DIR
        elif os.path.isabs(folder):
            out_dir = folder
        else:
            out_dir = os.path.join(_OUTPUT_DIR, folder)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    # ------------------------------------------------------------------
    @staticmethod
    def _build_content(prompt, first_frame, last_frame, reference_images,
                       reference_video, reference_audio,
                       video_urls: str, audio_urls: str) -> list:
        content = []
        if prompt.strip():
            content.append({"type": "text", "text": prompt.strip()})

        n_images = 0
        if first_frame is not None:
            content.append(_image_entry(first_frame[0], "first_frame"))
            n_images += 1
        if last_frame is not None:
            if first_frame is None:
                raise ValueError("[Mari Seedance2] last_frame requires first_frame")
            content.append(_image_entry(last_frame[0], "last_frame"))
            n_images += 1
        if reference_images is not None:
            for i in range(reference_images.shape[0]):
                if n_images >= _MAX_IMAGES:
                    print(f"[Mari Seedance2] ⚠ image limit reached ({_MAX_IMAGES}), "
                          f"dropping {reference_images.shape[0] - i} reference image(s)")
                    break
                content.append(_image_entry(reference_images[i], "reference_image"))
                n_images += 1

        # video refs: VIDEO input first, then URLs (API cap: 3 total)
        videos = []
        if reference_video is not None:
            videos.append(_video_to_data_url(reference_video))
        videos += [u.strip() for u in video_urls.split(",") if u.strip()]
        for url in videos[:3]:
            content.append({"type": "video_url", "video_url": {"url": url},
                            "role": "reference_video"})

        # audio refs: AUDIO input first, then URLs (API cap: 3 total)
        audios = []
        if reference_audio is not None:
            audios.append(_audio_to_data_url(reference_audio))
        audios += [u.strip() for u in audio_urls.split(",") if u.strip()]
        for url in audios[:3]:
            content.append({"type": "audio_url", "audio_url": {"url": url},
                            "role": "reference_audio"})

        if audios and n_images == 0 and not videos:
            print("[Mari Seedance2] ⚠ audio-only reference: the API requires audio to "
                  "accompany at least one image or video reference — this may be rejected")

        if not content:
            raise ValueError("[Mari Seedance2] empty request — provide a prompt and/or images")
        return content

    # ------------------------------------------------------------------
    @staticmethod
    def _poll(base: str, headers: dict, task_id: str, ssl_verify: bool = True,
              node_id=None) -> dict:
        url = f"{base}/contents/generations/tasks/{task_id}"
        start = time.time()
        last_status = ""
        while True:
            _interrupt_check()
            elapsed = time.time() - start
            if elapsed > _MAX_WAIT:
                raise TimeoutError(f"[Mari Seedance2] task {task_id} still '{last_status}' "
                                   f"after {int(elapsed)}s — giving up")
            r = requests.get(url, headers=headers, timeout=_TIMEOUT_HTTP, verify=ssl_verify)
            if r.status_code != 200:
                raise RuntimeError(f"[Mari Seedance2] status check failed — {_api_error(r)}")
            data = r.json()
            status = data.get("status", "")
            if status != last_status:
                print(f"[Mari Seedance2] [{int(elapsed)}s] status: {status}")
                last_status = status
            if status == "succeeded":
                _progress_text(f"⏱ {int(elapsed)}s | succeeded — downloading...", node_id)
                return data
            if status in ("failed", "expired", "cancelled"):
                err = data.get("error", {})
                raise RuntimeError(f"[Mari Seedance2] task {status} — "
                                   f"{err.get('code', '')}: {err.get('message', data)}")
            # interrupt-friendly sleep with a live elapsed counter on the node
            for _ in range(_POLL_INTERVAL):
                _interrupt_check()
                _progress_text(f"⏱ {int(time.time() - start)}s | {status}", node_id)
                time.sleep(1)


NODE_CLASS_MAPPINGS = {
    "MariSeedance2": MariSeedance2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MariSeedance2": "Mari - Seedance 2.0 Video",
}
