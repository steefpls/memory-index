"""Download the EmbeddingGemma-300m ONNX q8 model into data/embeddinggemma_onnx.

Pulls from the ungated onnx-community mirror, so no HF login is needed.
The q8 (dynamic int8) variant is deliberate: EmbeddingGemma activations are
documented broken under fp16, and fp32 is 1.2GB for no measurable quality gain.

Usage: python scripts/download_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMBED_ONNX_REPO, EMBED_ONNX_DIR, EMBED_ONNX_FILENAME

# (repo path, local filename) — the .onnx_data name must stay untouched because
# the model proto references it by that exact relative name.
_FILES = [
    (f"onnx/{EMBED_ONNX_FILENAME}", EMBED_ONNX_FILENAME),
    (f"onnx/{EMBED_ONNX_FILENAME}_data", f"{EMBED_ONNX_FILENAME}_data"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
    ("config.json", "config.json"),
]


def main() -> int:
    from huggingface_hub import hf_hub_download

    EMBED_ONNX_DIR.mkdir(parents=True, exist_ok=True)

    for repo_path, local_name in _FILES:
        dest = EMBED_ONNX_DIR / local_name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[SKIP] {local_name} already present ({dest.stat().st_size:,} bytes)")
            continue
        print(f"[GET]  {repo_path} ...")
        cached = hf_hub_download(EMBED_ONNX_REPO, repo_path)
        dest.write_bytes(Path(cached).read_bytes())
        print(f"[OK]   {local_name} ({dest.stat().st_size:,} bytes)")

    print(f"\nModel ready in {EMBED_ONNX_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
