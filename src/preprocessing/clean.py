import pandas as pd
import re, html
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "twcs.csv"

def load_raw(path: str):
    cols = pd.read_csv(path, usecols=['tweet_id', 'inbound', 'created_at', 'text'])
    return cols

def filter_airlines(df):
    user_tweets = df['inbound'] == True
    airways_mask = df['text'].str.contains("@Delta|@AmericanAir|@British_Airways", case=False)
    combined = user_tweets & airways_mask
    return df[combined]

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def drop_junk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text_clean"])
    df = df[df["text_clean"].str.split().str.len() >= 4]
    df = df.drop_duplicates(subset="text_clean")
    return df

def main():
    df = load_raw(RAW_FILE)
    print("loaded:", df.shape)

    df = filter_airlines(df)
    print("filtered:", df.shape)

    df = df.rename(columns={"text": "text_raw"})
    df['text_clean'] = df['text_raw'].apply(clean_text)
    
    df = drop_junk(df)
    print("cleaned:", df.shape)

    df = df.sample(n=6000, random_state=42)
    print("sampled:", df.shape)

    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tickets_clean.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print("saved to:", out_file)

if __name__ == '__main__': main()