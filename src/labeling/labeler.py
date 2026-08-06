import os
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from schema import Intent, Urgency, TicketLabel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = (PROJECT_ROOT / "docs" / "taxonomy.md").read_text(encoding="utf-8")

SYSTEM_PROMPT = f"""You are a triage assistant for an airline's customer support.
Classify the customer tweet you receive, using the taxonomy below.

{TAXONOMY}

Respond with a json object only, no other text, in exactly this format:
{{"intent": "lost_luggage", "urgency": "medium", "abusive": false}}

"intent" must be exactly one of: {", ".join(get_args(Intent))}
"urgency" must be exactly one of: {", ".join(get_args(Urgency))}
"abusive" must be a json boolean.
"""

load_dotenv()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

def label_tweet(text: str, max_retries: int = 3) -> TicketLabel:
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            max_tokens=200,
            temperature=0.6 * attempt,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        raw = response.choices[0].message.content
        try:
            return TicketLabel.model_validate_json(raw)
        except ValidationError:
            reason = response.choices[0].finish_reason
            print(f"  retry {attempt + 1}: finish_reason={reason}, response: {raw!r}")
    raise RuntimeError(f"failed after {max_retries} retries: {text[:60]}")

if __name__ == "__main__":
    tests = [
        "my bags are lost and I have no way to start getting them rerouted",
        "Thanks for the reply. We've got it sorted now.",
        "hello can i get a free flight to london rn",
    ]
    for t in tests:
        print(t[:50], "->", label_tweet(t))