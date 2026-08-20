from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD = PROJECT_ROOT / "data" / "processed" / "gold_final.csv"

def main():
    df = pd.read_csv(GOLD)
    n = len(df)
    print(f"gold set: {n} tweets\n")

    for field in ["intent", "urgency", "abusive"]:
        agree = (df[f"{field}_model"] == df[f"{field}_human"]).sum()
        print(f"{field:<10} {agree}/{n} = {agree / n:.1%}")

    print("\nintent agreement ανά κλάση (βάση: ανθρώπινη ετικέτα)")
    for cls, group in df.groupby("intent_human"):
        agree = (group["intent_model"] == group["intent_human"]).sum()
        print(f" {cls:<32} {agree}/{len(group)} = {agree / len(group):.0%}")

if __name__ == "__main__":
    main()