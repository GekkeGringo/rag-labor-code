import glob
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = glob.glob("/kaggle/input/**/contexts_dev.json", recursive=True)[0]
data = json.load(open(path, encoding="utf-8"))

MODEL = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "11GiB", 1: "11GiB"},
)
print("модель загружена, вопросов:", len(data))

SYSTEM = (
    "Ты помощник по Трудовому кодексу РФ. Отвечай на вопрос ТОЛЬКО на основе "
    "приведенных фрагментов кодекса, на русском языке, кратко (до 5 предложений). "
    "После каждого утверждения указывай статью в формате [ст. 123]. "
    "Если во фрагментах нет ответа на вопрос, напиши ровно: "
    "'В найденных статьях ответа нет.' и ничего больше. Ничего не придумывай."
)

results = []
for item in data:
    ctx = "\n\n---\n\n".join(c["text"] for c in item["contexts"])
    user_text = f"Фрагменты:\n\n{ctx}\n\nВопрос: {item['question']}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results.append({"id": item["id"], "answer": answer})
    print(item["id"], "|", answer[:200].replace("\n", " "))
    torch.cuda.empty_cache()

with open("/kaggle/working/answers_dev.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("=== ANSWERS JSON ===")
print(json.dumps(results, ensure_ascii=False))