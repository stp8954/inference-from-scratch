import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
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

with torch.inference_mode():
    for _ in range(32):
        scores = model(input_ids=ids, use_cache=False).logits[:, -1, :]
        next_id = scores.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() in stop_ids:
            break

answer_ids = ids[0, prompt_length:]
print(tokenizer.decode(answer_ids, skip_special_tokens=True))
