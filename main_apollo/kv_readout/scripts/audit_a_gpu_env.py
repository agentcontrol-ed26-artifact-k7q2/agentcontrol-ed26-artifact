"""A-GPU kill-switch — environment audit.

Read-only. Does NOT install packages, NOT download models. Reports whether
local GPU is usable and whether a GPU job would have to run on [CLUSTER_A].
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APOLLO = REPO / "main_apollo"


def _try(cmd, timeout=8):
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
        "GPU_AVAILABLE_locally": False,
        "torch": {},
        "transformers_installed": False,
        "vllm_installed": False,
        "disk_free_gb": None,
        "hf_cache": os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE"),
        "approval_env_var": "AGENTCONTROL_APOLLO_GPU_APPROVED",
        "approval_set": os.environ.get("AGENTCONTROL_APOLLO_GPU_APPROVED") == "1",
    }
    rc, txt = _try(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"])
    if rc == 0 and txt.strip():
        out["GPU_AVAILABLE_locally"] = True
        out["gpu_info"] = txt.strip()
    else:
        out["gpu_info"] = txt.strip()[:200] if txt else ""
    try:
        import torch
        out["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception as e:
        out["torch"] = {"installed": False, "error": str(e)[:200]}
    for pkg in ("transformers", "vllm"):
        try:
            __import__(pkg)
            out[f"{pkg}_installed"] = True
        except Exception:
            pass
    try:
        out["disk_free_gb"] = round(shutil.disk_usage(str(REPO)).free / (1024 ** 3), 2)
    except Exception:
        pass

    # Effective tier: local GPU usable for LLM serving requires (a) torch.cuda
    # available, (b) ≥ 14 GB VRAM (Qwen-7B fp16 minimum).
    eff_usable = bool(out["torch"].get("cuda_available"))
    has_vram = False
    if out["GPU_AVAILABLE_locally"] and out.get("gpu_info"):
        try:
            mib = int(out["gpu_info"].splitlines()[0].split(",")[1].strip().split()[0])
            has_vram = mib >= 14000
        except Exception:
            pass
    if not (eff_usable and has_vram):
        out["effective_tier"] = "TIER0_NO_LOCAL_GPU_USE_ORCD"
        out["effective_reason"] = (
            f"torch.cuda_available={out['torch'].get('cuda_available')}, "
            f"VRAM≥14GB={has_vram}. Must use [CLUSTER_A] H200."
        )
    else:
        out["effective_tier"] = "TIER1_LOCAL_GPU_USABLE"

    out_path = APOLLO / "kv_readout" / "experiments" / "a_gpu_env_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# A_GPU_ENV_AUDIT\n",
          f"\n- platform: {out['platform']}\n",
          f"- local nvidia-smi: {out['GPU_AVAILABLE_locally']}\n",
          f"- gpu_info: `{(out.get('gpu_info') or '')[:120]}`\n",
          f"- torch: {out['torch']}\n",
          f"- transformers installed: {out['transformers_installed']}\n",
          f"- vllm installed: {out['vllm_installed']}\n",
          f"- disk_free_gb: {out['disk_free_gb']}\n",
          f"- approval set: **{out['approval_set']}**\n",
          f"\n## Effective tier: **{out['effective_tier']}**\n",
          "\n" + out.get("effective_reason", "") + "\n",
          "\n## Implications\n\n"]
    if out["effective_tier"] == "TIER0_NO_LOCAL_GPU_USE_ORCD":
        md.append("Local GPU unusable for Qwen-7B-Instruct serving. The hidden-state probe "
                  "must run on [CLUSTER_A] H200. User has confirmed [CLUSTER_A] plink session is open. "
                  "Proceed only if `AGENTCONTROL_APOLLO_GPU_APPROVED=1`.\n")
    else:
        md.append("Local GPU usable. Probe can run locally without cluster job.\n")
    (APOLLO / "kv_readout" / "reports" / "A_GPU_ENV_AUDIT.md").write_text("".join(md), encoding="utf-8")
    print(f"effective_tier: {out['effective_tier']}")
    print(f"approval set: {out['approval_set']}")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
