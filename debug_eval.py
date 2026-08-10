"""Debug script: test what Qwen generates with chat template.
Run this inside your working environment:
    python debug_eval.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token

# Load model in fp16 on single GPU (no device_map)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=None,
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Test 1: basic classification
messages = [
    {"role": "system", "content": "Classify the sentiment of the following movie review as 'positive' or 'negative'."},
    {"role": "user", "content": "Text: This movie was absolutely fantastic!"},
]
prompt = tok.apply_chat_template(messages, tokenize=False)
print("=" * 60)
print("TEST 1: Basic classification")
print(f"Prompt: {repr(prompt[:200])}")

tok.padding_side = "left"
encoded = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=256)
tok.padding_side = "right"

device = "cuda" if torch.cuda.is_available() else "cpu"
encoded = {k: v.to(device) for k, v in encoded.items()}

with torch.no_grad():
    outputs = model.generate(
        **encoded,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )

input_len = encoded["input_ids"].shape[1]
gen = outputs[0, input_len:]
pred = tok.decode(gen, skip_special_tokens=True).strip().lower()
print(f"Generated tokens: {gen.tolist()}")
print(f"Decoded: {repr(pred)}")
print(f"Match 'positive': {pred == 'positive'}")

# Test 2: all 4 tasks with the exact eval format
print()
print("=" * 60)
print("TEST 2: All 4 tasks")
test_examples = [
    ("sentiment", "The acting was superb.", "positive"),
    ("topic", "The quarterback threw a touchdown pass.", "sports"),
    ("question_type", "Is Paris the capital of France?", "yes_no"),
    ("toxicity", "You are an idiot.", "toxic"),
]

instructions = {
    "sentiment": "Classify the sentiment of the following movie review as 'positive' or 'negative'.",
    "topic": "Classify whether the following sentence is about 'sports' or 'technology'.",
    "question_type": "Classify the following question as 'yes_no' or 'factual'.",
    "toxicity": "Classify the following text as 'toxic' or 'safe'.",
}

correct = 0
total = 0
for domain, text, expected in test_examples:
    msgs = [
        {"role": "system", "content": instructions[domain]},
        {"role": "user", "content": f"Text: {text}"},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False)

    tok.padding_side = "left"
    encoded = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    tok.padding_side = "right"

    with torch.no_grad():
        outputs = model.generate(
            **encoded, max_new_tokens=8, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )

    gen = outputs[0, encoded["input_ids"].shape[1]:]
    pred = tok.decode(gen, skip_special_tokens=True).strip().lower()
    is_match = pred == expected.lower()
    print(f"  {domain:>15} | expected={expected:<10} | got={repr(pred):<20} | match={is_match}")
    if is_match:
        correct += 1
    total += 1

print(f"  Total: {correct}/{total} correct")
