"""Phase 0: GPU / local-inference environment audit.

Read-only. Does NOT install packages, NOT download models.
Decides the rescue tier:
  TIER0_NO_GPU                    no usable local GPU; API-only path
  TIER1_SINGLE_GPU                one GPU, can host a small/medium open-weight model
  TIER2_MULTI_GPU                 multi-GPU, can host larger models
  TIER3_CONTROLLER_TRAINING_OK    only after oracle gap is proven; never set automatically
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT_JSON = Path(__file__).resolve().parent.parent / "experiments" / "gpu_env_audit.json"
OUT_MD = Path(__file__).resolve().parent.parent / "reports" / "GPU_ENVIRONMENT_AUDIT.md"


def _try(cmd: list[str], timeout: int = 8) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return -1, "not found"
    except Exception as e:
        return -2, str(e)


def main() -> int:
    out = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "node": platform.node(),
        "GPU_AVAILABLE": False,
        "gpu_count": 0,
        "gpu_info": "",
        "torch": {},
        "transformers_installed": False,
        "vllm_installed": False,
        "accelerate_installed": False,
        "disk_free_gb": None,
        "model_cache_paths": {},
    }

    rc, txt = _try(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"])
    if rc == 0 and txt.strip():
        out["GPU_AVAILABLE"] = True
        out["gpu_info"] = txt.strip()
        out["gpu_count"] = len([l for l in txt.strip().splitlines() if l.strip()])
    else:
        out["gpu_info"] = txt.strip()[:200] if txt else ""

    try:
        import torch  # type: ignore
        out["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
        if out["torch"]["cuda_available"]:
            out["GPU_AVAILABLE"] = True
            out["gpu_count"] = max(out["gpu_count"], out["torch"]["cuda_device_count"])
    except Exception as e:
        out["torch"] = {"installed": False, "error": str(e)[:200]}

    for pkg in ("transformers", "vllm", "accelerate"):
        try:
            __import__(pkg)
            out[f"{pkg}_installed"] = True
        except Exception:
            pass

    try:
        usage = shutil.disk_usage(str(REPO))
        out["disk_free_gb"] = round(usage.free / (1024**3), 2)
    except Exception:
        pass

    for env in ("HF_HOME", "TRANSFORMERS_CACHE", "OLLAMA_MODELS"):
        v = os.environ.get(env)
        if v:
            out["model_cache_paths"][env] = v

    # Effective tier — accounts for whether torch can actually use the GPU and
    # whether VRAM is large enough to host a useful LLM (~6 GB fp16 minimum for 3B,
    # 14 GB for 7B). nvidia-smi may detect a card that torch cannot use.
    effective_gpu_usable = bool(out["torch"].get("cuda_available"))
    has_vram_for_llm = False
    if out["GPU_AVAILABLE"] and out["gpu_info"]:
        # Parse "Name, total MiB, free MiB, driver"
        try:
            mib = int(out["gpu_info"].splitlines()[0].split(",")[1].strip().split()[0])
            has_vram_for_llm = mib >= 8000  # ≥ 8 GB to host even a 3B fp16
        except Exception:
            pass
    if not (effective_gpu_usable and has_vram_for_llm):
        out["recommended_tier"] = "TIER0_NO_GPU"
        out["effective_gpu_usable"] = False
        out["effective_reason"] = (
            "GPU detected by nvidia-smi but unusable for LLM serving in this run: "
            f"torch.cuda_available={out['torch'].get('cuda_available')}, "
            f"VRAM≥8GB={has_vram_for_llm}. API-only path."
        )
    elif out["gpu_count"] == 1:
        out["recommended_tier"] = "TIER1_SINGLE_GPU"
        out["effective_gpu_usable"] = True
    else:
        out["recommended_tier"] = "TIER2_MULTI_GPU"
        out["effective_gpu_usable"] = True

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# GPU_ENVIRONMENT_AUDIT\n",
          f"\n- platform: {out['platform']}\n",
          f"- python: {out['python_version']}\n",
          f"- GPU_AVAILABLE: **{out['GPU_AVAILABLE']}**\n",
          f"- gpu_count: {out['gpu_count']}\n",
          f"- gpu_info: `{out['gpu_info'][:150]}`\n",
          f"- torch: {out['torch']}\n",
          f"- transformers installed: {out['transformers_installed']}\n",
          f"- vllm installed: {out['vllm_installed']}\n",
          f"- accelerate installed: {out['accelerate_installed']}\n",
          f"- disk_free_gb: {out['disk_free_gb']}\n",
          f"- model_cache_paths: {out['model_cache_paths']}\n",
          f"\n## Recommended tier: **{out['recommended_tier']}**\n",
          "\n## Implications\n\n"]
    if out["recommended_tier"] == "TIER0_NO_GPU":
        md.append("No usable local GPU detected. The Main Rescue Fork will run on cached real-API outcomes "
                  "(DeepSeek + Together AI). Local model serving is **not** authorized in this run. "
                  "The interactive task pool can still be evaluated end-to-end via API "
                  "(observations like `run_tests` and `retrieve` are local + free; only `cheap_*` and `strong_*` "
                  "actions go through the API). **Do not download large models.** **Do not install vllm.**\n")
    else:
        md.append("GPU detected. Local model serving is possible but is NOT authorized automatically. "
                  "Ask the user before downloading any model > 1 GB or installing vllm. "
                  "Until oracle graph headroom is proven, do NOT train any controller / readout.\n")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(f"GPU_AVAILABLE={out['GPU_AVAILABLE']}; tier={out['recommended_tier']}")
    print(f"wrote {OUT_JSON.relative_to(REPO)}, {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
