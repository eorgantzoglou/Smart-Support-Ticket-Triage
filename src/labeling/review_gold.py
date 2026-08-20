from pathlib import Path
from typing import get_args

import pandas as pd
from schema import Intent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE = PROJECT_ROOT / "data" / "processed" / "gold_review.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "gold_final.csv"

INTENTS = list(get_args(Intent))
URGENCIES = {"h": "high", "m": "medium", "l": "low"}
LEGEND = "\n".join(f"  {i} = {name}" for i, name in enumerate(INTENTS, start=1))

def apply_commands(raw: str, label: dict):
    for token in raw.split():
        if token[0] == "i" and token[1:].isdigit() and 1 <= int(token[1:]) <= len(INTENTS):
            label["intent"] = INTENTS[int(token[1:]) - 1]
        elif token[0] == "u" and token[1:] in URGENCIES:
            label["urgency"] = URGENCIES[token[1:]]
        elif token == "a":
            label["abusive"] = not label["abusive"]
        else:
            return None
    return label

def main():
    df = pd.read_csv(IN_FILE)

    done_ids = set()
    if OUT_FILE.exists():
        done_ids = set(pd.read_csv(OUT_FILE, usecols=["tweet_id"])["tweet_id"])

    todo = df[~df["tweet_id"].isin(done_ids)]
    print(f"reviewed: {len(done_ids)}/{len(df)} - απομένουν {len(todo)}")
    print("Enter=σωστό | i<νούμερο> | uh/um/ul | a | q=έξοδος | ?=κλάσεις")

    for count, row in enumerate(todo.itertuples(), start=1):
        model = {"intent": row.intent, "urgency": row.urgency, "abusive": row.abusive}
        print("\n" + "=" * 70)
        print(row.text_clean)
        print("-" * 70)
        print(f"[{count}/{len(todo)}]  {model['intent']} | {model['urgency']} | abusive={model['abusive']}")

        while True:
            raw = input("> ").strip().lower()
            if raw == "q":
                return
            if raw == "?":
                print(LEGEND)
                continue
            final = apply_commands(raw, dict(model))
            if final is not None:
                break
            print("δεν το κατάλαβα - πάτα ? για τη λίστα")

        record = {
            "tweet_id": row.tweet_id,
            "text_clean": row.text_clean,
            "intent_model": model["intent"],
            "urgency_model": model["urgency"],
            "abusive_model": model["abusive"],
            "intent_human": final["intent"],
            "urgency_human": final["urgency"],
            "abusive_human": final["abusive"],
        }
        pd.DataFrame([record]).to_csv(
            OUT_FILE, mode="a", header=not OUT_FILE.exists(),
            index=False, encoding="utf-8-sig",
        )

    print("\nΤέλος - όλα ελεγμένα!")


if __name__ == "__main__":
    main()