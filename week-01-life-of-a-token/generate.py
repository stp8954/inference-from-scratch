import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import try_to_load_from_cache

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--report", type=Path, help="write a machine-readable run report")
args = parser.parse_args()
device = torch.device(args.device)
if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is not available")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float32
).to(device).eval()
model_device = next(model.parameters()).device
if model_device.type != device.type or (device.index is not None and model_device.index != device.index):
    raise RuntimeError(f"model is on {model_device}, expected {device}")

messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
ids = tokenizer(
    prompt, add_special_tokens=False, return_tensors="pt"
)["input_ids"]
ids = ids.to(device)
if ids.device.type != device.type or (device.index is not None and ids.device.index != device.index):
    raise RuntimeError(f"inputs are on {ids.device}, expected {device}")
prompt_length = ids.shape[1]

stop_ids = model.generation_config.eos_token_id
if stop_ids is None:
    stop_ids = []
elif isinstance(stop_ids, int):
    stop_ids = [stop_ids]

iteration_input_lengths = []
stop_reason = "max_new_tokens"
with torch.inference_mode():
    for _ in range(32):
        iteration_input_lengths.append(ids.shape[1])
        scores = model(input_ids=ids, use_cache=False).logits[:, -1, :]
        next_id = scores.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() in stop_ids:
            stop_reason = "eos"
            break

answer_ids = ids[0, prompt_length:]
answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
print(answer_text)

if args.report:
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cached_config = try_to_load_from_cache(model_name, "config.json")
    model_revision = getattr(model.config, "_commit_hash", None)
    if model_revision is None and cached_config:
        model_revision = Path(cached_config).parent.name
    report = {
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "model_revision": model_revision,
        "device": str(model_device),
        "input_device": str(ids.device),
        "precision": "float32",
        "prompt_token_count": prompt_length,
        "generated_token_count": int(answer_ids.shape[0]),
        "generated_token_count_includes_eos": True,
        "final_generated_token_id": int(answer_ids[-1].item()) if answer_ids.numel() else None,
        "generated_text": answer_text,
        "displayed_text": answer_text,
        "loop_iterations": len(iteration_input_lengths),
        "stop_reason": stop_reason,
        "max_new_tokens": 32,
        "iteration_input_sequence_lengths": iteration_input_lengths,
        "source_sha256": source_hash,
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    if report["generated_token_count"] != report["loop_iterations"]:
        raise RuntimeError("generated token count must equal loop iterations")
    if len(iteration_input_lengths) != report["generated_token_count"]:
        raise RuntimeError("one input length is required per generated token")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
