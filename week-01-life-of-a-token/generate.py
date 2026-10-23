import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float32
).to("cpu").eval()

messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
ids = tokenizer(
    prompt, add_special_tokens=False, return_tensors="pt"
)["input_ids"]
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
