import time
from pathlib import Path

import pandas as pd
from labeler import label_tweet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE = PROJECT_ROOT / "data" / "processed" / "tickets_clean.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "pilot_labels.csv"

def main():
    df = pd.read_csv(IN_FILE)
    pilot = df.sample(n=50, random_state=42)

    rows = []
    start = time.perf_counter()
    for i, tweet in enumerate(pilot["text_clean"], start=1):
        label = label_tweet(tweet)
        rows.append(label.model_dump())
        print(f"{i}/{len(pilot)} labeled", end="\r")
    elapsed = time.perf_counter() - start

    labels_df = pd.DataFrame(rows, index=pilot.index)
    out = pd.concat([pilot, labels_df], axis=1)

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nlabeled {len(out)} tweets in {elapsed:.1f}s "
          f"({elapsed / len(out):.2f}s per tweet)")
    print(out["intent"].value_counts())


if __name__ == "__main__":
    main()