import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELED = PROJECT_ROOT / "data" / "processed" / "tickets_labeled.csv"
GOLD = PROJECT_ROOT / "data" / "processed" / "gold_final.csv"
OUT_DIR  = PROJECT_ROOT / "data" / "training"

SYSTEM_PROMPT = (
    "You are a triage assistant for an airline's customer support. "
    "Classify the customer tweet. Respond with json only, in exactly this format: "
    '{"intent": "...", "urgency": "...", "abusive": true/false}. '
    "intent must be one of: delay_disruption, checkin_boarding_issue, "
    "flight_cancellation_rebooking, lost_luggage, special_assistance, "
    "general_complaint, general_question, praise_feedback, spam_irrelevant, "
    "other_unclear. urgency must be one of: high, medium, low."
)

def to_example(row):
    label = {"intent": row.intent, "urgency": row.urgency, "abusive": bool(row.abusive)}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row.text_clean},
            {"role": "assistant", "content": json.dumps(label)},
        ]
    }

def write_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in df.itertuples():
            f.write(json.dumps(to_example(row), ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(df)} examples")

def main():
    df = pd.read_csv(LABELED)
    gold_ids = set(pd.read_csv(GOLD, usecols=["tweet_id"])["tweet_id"])

    pool = df[~df["tweet_id"].isin(gold_ids)]
    print(f"labeled: {len(df)}  gold (excluded):  {len(gold_ids)}  pool: {len(pool)}")

    val = pool.sample(frac=0.1, random_state=42)
    train = pool.drop(val.index)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, OUT_DIR / "train.jsonl")
    write_jsonl(val, OUT_DIR / "val.jsonl")

if __name__ == "__main__":
    main()