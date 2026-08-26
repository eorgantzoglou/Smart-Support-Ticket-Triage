import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_ID = "eorgantzoglou/qwen2.5-0.5b-airline-triage-lora"

SYSTEM_PROMPT = (
    "You are a triage assistant for an airline's customer support. "
    "Classify the customer tweet. Respond with json only, in exactly this format: "
    '{"intent": "...", "urgency": "...", "abusive": true/false}. '
    "intent must be one of: delay_disruption, checkin_boarding_issue, "
    "flight_cancellation_rebooking, lost_luggage, special_assistance, "
    "general_complaint, general_question, praise_feedback, spam_irrelevant, "
    "other_unclear. urgency must be one of: high, medium, low."
)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ADAPTER_ID)
    model = model.merge_and_unload()
    model.eval()
    return tokenizer, model

def triage_text(text: str, tokenizer, model, max_new_tokens: int = 60)-> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


if __name__ == "__main__":
    import time

    print("loading model...")
    tokenizer, model = load_model()

    tests = [
        "my bags are lost and I have no way to start getting them rerouted",
        "Thanks for the reply. We've got it sorted now.",
        "hello can i get a free flight to london rn",
    ]
    for t in tests:
        start = time.perf_counter()
        result = triage_text(t, tokenizer, model)
        print(f"{t[:50]} -> {result}  ({time.perf_counter() - start:.1f}s)")      