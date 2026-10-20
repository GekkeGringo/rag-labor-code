import glob
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = glob.glob("/kaggle/input/**/contexts_test.json", recursive=True)[0]
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

SYSTEM_RAG = (
    "Ты помощник по Трудовому кодексу РФ. Отвечай на вопрос ТОЛЬКО на основе "
    "приведенных фрагментов кодекса, на русском языке, кратко (до 5 предложений). "
    "После каждого утверждения указывай статью в формате [ст. 123]. "
    "Если во фрагментах нет ответа на вопрос, напиши ровно: "
    "'В найденных статьях ответа нет.' и ничего больше. Ничего не придумывай."
)
SYSTEM_CLOSED = (
    "Ты помощник по Трудовому кодексу РФ. Ответь на вопрос на русском языке, "
    "кратко (до 5 предложений). Если помнишь статью Трудового кодекса, укажи ее "
    "в формате [ст. 123]. Если в Трудовом кодексе ответа на вопрос нет или ты его "
    "не знаешь, напиши ровно: 'В Трудовом кодексе ответа нет.' и ничего больше. "
    "Ничего не придумывай."
)


def ctx_text(contexts):
    return "\n\n---\n\n".join(c["text"] for c in contexts)


def build(mode, item):
    q = item["question"]
    if mode == "rag":
        return SYSTEM_RAG, f"Фрагменты:\n\n{ctx_text(item['contexts'])}\n\nВопрос: {q}"
    if mode == "oracle":
        if "oracle_contexts" not in item:
            return None
        return SYSTEM_RAG, f"Фрагменты:\n\n{ctx_text(item['oracle_contexts'])}\n\nВопрос: {q}"
    return SYSTEM_CLOSED, f"Вопрос: {q}"

r_rag = run("rag")
r_oracle = run("oracle")
r_closed = run("closed")