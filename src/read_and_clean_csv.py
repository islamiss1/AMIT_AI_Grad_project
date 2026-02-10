import pandas as pd
import re

# ----------------------------
# Text cleaning function
# ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    # Remove HTML tags (e.g. <P>, </P>)
    text = re.sub(r"<[^>]+>", "", text)

    # Fix spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Fix spaces before apostrophes (firm 's -> firm's)
    text = re.sub(r"\s+'", "'", text)

    # Normalize multiple spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


# ----------------------------
# Main pipeline
# ----------------------------
def csv_to_clean_json(
    csv_path: str,
    json_path: str,
    max_rows: int = 5000
):
    # 1. Read CSV
    df = pd.read_csv(csv_path)

    # 2. Keep only required columns
    df = df[["question", "long_answers", "short_answers"]]

    # 3. Use first 1000 rows
    df = df.head(max_rows)

    # 4. Drop rows with any empty cell
    df = df.dropna()
    df = df[(df != "").all(axis=1)]

    # 5. Clean text columns
    for col in ["question", "long_answers", "short_answers"]:
        df[col] = df[col].apply(clean_text)

    # 6. Save as JSON
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    print(f"✅ Saved {len(df)} cleaned rows to {json_path}")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    csv_to_clean_json(
        csv_path= r"D:\project islam\data\raw\Natural-Questions-Base.csv",
        json_path=r"D:\project islam\data\preprocessed\cleaned_data.json"
    )
