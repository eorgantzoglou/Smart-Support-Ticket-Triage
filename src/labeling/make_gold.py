from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE = PROJECT_ROOT / "data" / "processed" / "tickets_labeled.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "gold_review.csv"

df = pd.read_csv(IN_FILE)

gold = df.sample(n=300, random_state=7)
gold = gold.sort_values("intent")

gold["human_intent"] = ""
gold["human_urgency"] = ""
gold["human_abusive"] = ""

gold.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print("saved:", len(gold), "tweets to", OUT_FILE)