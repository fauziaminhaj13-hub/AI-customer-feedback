import re

import pandas as pd


FEEDBACK_COLUMN_PRIORITY = (
    "feedback",
    "review",
    "reviews",
    "text",
    "tweettext",
    "fulltext",
    "content",
    "comment",
    "message",
)
DATE_COLUMN_PRIORITY = (
    "createdat",
    "publishedat",
    "updatedat",
    "timestamp",
    "datetime",
    "date",
    "time",
)
DATE_COLUMN_MARKERS = (
    "created",
    "date",
    "datetime",
    "published",
    "timestamp",
    "updated",
)
MINIMUM_DATE_PARSE_RATIO = 0.8


def normalize_column_name(column):
    return re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())


def find_feedback_column(df):
    normalized_columns = {
        normalize_column_name(column): column for column in df.columns
    }
    for candidate in FEEDBACK_COLUMN_PRIORITY:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    for column in df.columns:
        if df[column].dtype == "object":
            return column

    if len(df.columns) == 0:
        return None
    return df.columns[0]


def remove_blank_feedback(df, feedback_column):
    clean_df = df.dropna(subset=[feedback_column]).copy()
    clean_df[feedback_column] = clean_df[feedback_column].astype(str).str.strip()
    return clean_df[clean_df[feedback_column] != ""]


def find_date_column(df, excluded_column=None):
    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
        if column != excluded_column
    }
    candidates = []
    for name in DATE_COLUMN_PRIORITY:
        column = normalized_columns.get(name)
        if column is not None and column not in candidates:
            candidates.append(column)

    for normalized_name, column in normalized_columns.items():
        if any(marker in normalized_name for marker in DATE_COLUMN_MARKERS):
            if column not in candidates:
                candidates.append(column)

    for column in candidates:
        values = df[column].dropna()
        if values.empty:
            continue
        if pd.api.types.is_numeric_dtype(values):
            continue
        parsed = values.map(lambda value: pd.to_datetime(value, errors="coerce"))
        if parsed.notna().mean() >= MINIMUM_DATE_PARSE_RATIO:
            return column
    return None
