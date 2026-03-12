"""Export CodeRankEmbed to ONNX format with fp16 conversion for GPU acceleration."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

onnx_dir = "data/coderank_onnx"
onnx_path = os.path.join(onnx_dir, "model.onnx")

if os.path.exists(onnx_path):
    print("[OK] ONNX model already exported at", onnx_dir)
    sys.exit(0)

import torch
from sentence_transformers import SentenceTransformer

print("[INFO] Loading CodeRankEmbed model with PyTorch...")
t0 = time.time()
model = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True)
print(f"  Model loaded in {time.time()-t0:.1f}s")

transformer = model[0].auto_model
tokenizer = model[0].tokenizer

dummy = tokenizer("test", return_tensors="pt", padding=True, truncation=True, max_length=8192)

os.makedirs(onnx_dir, exist_ok=True)

print("[INFO] Exporting to ONNX format...")
t0 = time.time()
transformer.eval()
with torch.no_grad():
    torch.onnx.export(
        transformer,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=18,
    )
print(f"  Exported in {time.time()-t0:.1f}s")

tokenizer.save_pretrained(onnx_dir)

# Convert to float16 for faster inference and lower memory
print("[INFO] Converting to float16...")
t0 = time.time()
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

fp32_model = onnx.load(onnx_path, load_external_data=True)
fp16_model = convert_float_to_float16(fp32_model, keep_io_types=True)

# Clean save (delete old files first to avoid appending)
os.remove(onnx_path)
data_path = os.path.join(onnx_dir, "model.onnx.data")
if os.path.exists(data_path):
    os.remove(data_path)

onnx.save_model(
    fp16_model,
    onnx_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="model.onnx.data",
    size_threshold=1024,
)
print(f"  Converted in {time.time()-t0:.1f}s")

weights_mb = os.path.getsize(data_path) / 1024 / 1024
print(f"[OK] ONNX model exported to {onnx_dir} (fp16, {weights_mb:.0f}MB weights)")
