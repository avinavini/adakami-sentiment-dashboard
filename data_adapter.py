import re
from typing import Optional, Tuple

import pandas as pd


TEXT_COLUMNS = ("teks_ulasan", "review", "comment", "text", "tweet_text", "content", "body")
SENTIMENT_COLUMNS = ("sentimen_prediksi", "sentimen", "sentiment", "label", "sentiment_label")
MONTH_COLUMNS = ("bulan", "month", "created_at", "createdat", "timestamp", "date")

SENTIMENT_MAP = {
    "negative": "Negatif",
    "negatif": "Negatif",
    "neutral": "Netral",
    "netral": "Netral",
    "positive": "Positif",
    "positif": "Positif",
}


def _first_existing_column(df: pd.DataFrame, columns: Tuple[str, ...]) -> Optional[str]:
    normalized = {str(column).strip().casefold(): column for column in df.columns}
    for column in columns:
        if column in normalized:
            return normalized[column]
    return None


def _clean_text(value: object) -> str:
    text = str(value).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9A-ZÀ-ÿ]+", " ", text)).strip()


def _month_series(df: pd.DataFrame, column: str) -> pd.Series:
    normalized_column = str(column).strip().casefold()
    if normalized_column in {"bulan", "month"}:
        month = pd.to_numeric(df[column], errors="coerce")
        invalid = month.isna() | (month < 1) | (month > 12) | (month % 1 != 0)
        if invalid.any():
            raise ValueError("Kolom bulan harus berisi bilangan bulat dari 1 sampai 12.")
        return month.astype(int)

    date = pd.to_datetime(df[column], errors="coerce", utc=True)
    if date.isna().any():
        raise ValueError("Kolom tanggal memiliki nilai kosong atau tidak valid.")
    return date.dt.month.astype(int)


def normalize_external_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Map Xquik or social review CSV rows to the dashboard review schema."""
    text_column = _first_existing_column(df, TEXT_COLUMNS)
    if text_column is None:
        raise ValueError("CSV membutuhkan kolom teks seperti text, tweet_text, review, atau comment.")

    sentiment_column = _first_existing_column(df, SENTIMENT_COLUMNS)
    month_column = _first_existing_column(df, MONTH_COLUMNS)
    if month_column is None:
        raise ValueError(
            "CSV membutuhkan kolom waktu seperti created_at, timestamp, date, month, atau bulan."
        )

    text = df[text_column].fillna("").astype(str).str.strip()
    out = pd.DataFrame(index=df.index)
    out["teks_ulasan"] = text
    out = out[out["teks_ulasan"] != ""].copy()
    out["teks_bersih"] = out["teks_ulasan"].map(_clean_text)

    if sentiment_column:
        sentiment = (
            df.loc[out.index, sentiment_column]
            .fillna("Netral")
            .astype(str)
            .str.strip()
            .str.lower()
            .map(SENTIMENT_MAP)
            .fillna("Netral")
        )
    else:
        sentiment = pd.Series(["Netral"] * len(out), index=out.index)

    out["sentimen_prediksi"] = sentiment
    out["sentimen"] = sentiment
    out["rating"] = sentiment.map({"Positif": 5, "Netral": 3, "Negatif": 1}).astype(int)
    out["bulan"] = _month_series(df.loc[out.index], month_column)
    return out.reset_index(drop=True)
