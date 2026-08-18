import gc
import os
from typing import Optional


def normalize_device(device: Optional[str], fallback_gpu_id: int = 0, allow_auto: bool = False) -> str:
    raw = (device or "").strip().lower()
    if allow_auto and raw == "auto":
        return "auto"
    if raw in {"", "cuda", "gpu"}:
        return f"cuda:{fallback_gpu_id}" if _cuda_available() else "cpu"
    if raw.isdigit():
        return f"cuda:{raw}" if _cuda_available() else "cpu"
    if raw.startswith("cuda"):
        return raw if _cuda_available() else "cpu"
    return raw or "cpu"


def set_current_cuda_device(device: str, fallback_gpu_id: int = 0) -> None:
    if not str(device).startswith("cuda"):
        return
    import torch

    idx = fallback_gpu_id
    if ":" in str(device):
        idx = int(str(device).split(":", 1)[1])
    if torch.cuda.is_available():
        torch.cuda.set_device(idx)


def gpu_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def configure_huggingface_paths() -> None:
    # Keep HF caches stable for both CLI and imported use.
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
