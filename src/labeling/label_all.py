import time
from pathlib import Path
import pandas as pd
from labeler import label_tweet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE = PROJECT_ROOT / "data" / "processed" / "tickets_clean.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "tickets_labeled.csv"

def main():
    df = pd.read_csv(IN_FILE)

    done_ids = set()
    if OUT_FILE.exists():
        done_ids = set(pd.read_csv(OUT_FILE, usecols=["tweet_id"])["tweet_id"])
        print(f"resuming: {len(done_ids)} already labeled")

    todo = df[~df["tweet_id"].isin(done_ids)]
    print(f"to label: {len(todo)}")

    start = time.perf_counter()
    for count, row in enumerate(todo.itertuples(), start=1):
        try:
            label = label_tweet(row.text_clean)
        except RuntimeError as e:
            print(f"\nSKIPPED {row.tweet_id}: {e}")
            continue

        record = {
            "tweet_id": row.tweet_id,
            "created_at": row.created_at,
            "text_raw": row.text_raw,
            "text_clean": row.text_clean,
            **label.model_dump(),
        }
        pd.DataFrame([record]).to_csv(
            OUT_FILE, mode="a", header=not OUT_FILE.exists(),
            index=False, encoding="utf-8-sig",
        )

        if count % 25 == 0:
            rate = (time.perf_counter() - start) / count
            left = rate * (len(todo) - count) / 60
            print(f"{count}/{len(todo)}  ({rate:.2f}s/tweet, ~{left:.0f} min left)", end="\r")

if __name__ == "__main__":
    main()