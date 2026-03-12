"""Detect GPU vendor, recommend packages, and verify CUDA actually loads.

Modes:
  --install-hint   Print the pip package name for setup.bat.
  --check-cuda     Exit 0 if CUDAExecutionProvider actually loads, 1 if not.
  (default)        Print a human-readable status message.
"""

import subprocess
import sys


# CUDA 12 pip packages needed by onnxruntime-gpu when CUDA Toolkit isn't system-installed.
CUDA_PIP_PACKAGES = [
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
]


def detect_gpu_vendor() -> str:
    """Detect GPU vendor using PowerShell Get-CimInstance.

    Returns: 'nvidia', 'amd', 'intel', or 'unknown'.
    """
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        output_lower = output.lower()
        if "nvidia" in output_lower:
            return "nvidia"
        elif "amd" in output_lower or "radeon" in output_lower:
            return "amd"
        elif "intel" in output_lower:
            return "intel"
    except Exception:
        pass
    return "unknown"


def get_ort_package(vendor: str) -> str:
    """Return the correct onnxruntime pip package for the GPU vendor."""
    if vendor == "nvidia":
        return "onnxruntime-gpu"
    elif vendor in ("amd", "intel"):
        return "onnxruntime-directml"
    else:
        return "onnxruntime"


def _register_nvidia_dll_dirs():
    """Add nvidia pip-wheel DLL dirs to PATH so CUDA libraries are discoverable."""
    if sys.platform != "win32":
        return
    import os, pathlib
    nvidia_base = pathlib.Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages" / "nvidia"
    if not nvidia_base.exists():
        return
    for bin_dir in nvidia_base.glob("*/bin"):
        if bin_dir.is_dir() and any(bin_dir.glob("*.dll")):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def check_cuda_loads() -> bool:
    """Test whether CUDAExecutionProvider actually loads (not just listed).

    Creates a minimal ONNX session to verify CUDA libraries are present.
    Returns True if CUDA is functional.
    """
    _register_nvidia_dll_dirs()
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        # Try creating a session with CUDA — this is where missing DLLs cause failures
        import tempfile, os
        try:
            import onnx
            from onnx import helper, TensorProto
            # Minimal identity model
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
            node = helper.make_node("Identity", ["X"], ["Y"])
            graph = helper.make_graph([node], "test", [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            tmp = os.path.join(tempfile.gettempdir(), "_cuda_check.onnx")
            onnx.save(model, tmp)
            sess = ort.InferenceSession(tmp, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            active = sess.get_providers()
            os.unlink(tmp)
            return "CUDAExecutionProvider" in active
        except Exception:
            return False
    except ImportError:
        return False


def main():
    if "--install-hint" in sys.argv:
        vendor = detect_gpu_vendor()
        pkg = get_ort_package(vendor)
        print(pkg)
        return

    if "--check-cuda" in sys.argv:
        sys.exit(0 if check_cuda_loads() else 1)

    # Status mode: report from installed onnxruntime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            if check_cuda_loads():
                print("[OK] NVIDIA CUDA verified - GPU acceleration is working")
            else:
                print("[WARN] CUDA provider listed but failed to load - missing CUDA libraries?")
                print("       Run setup.bat to install CUDA pip packages automatically.")
        elif "DmlExecutionProvider" in providers:
            print("[OK] DirectML detected (AMD/Intel GPU) - will use GPU acceleration")
        else:
            print("[INFO] No GPU acceleration available - will use CPU")
        print("  Available providers:", providers)
    except ImportError:
        print("[WARN] onnxruntime not installed")


if __name__ == "__main__":
    main()
