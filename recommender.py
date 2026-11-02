"""
Content-Based Movie Recommender - core logic
Dataset: TMDB 5000 Movie Dataset (Kaggle)
Link   : https://www.kaggle.com/datasets/tmdb/tmdb-5000-movie-dataset
File   : tmdb_5000_movies.csv
"""

import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "tmdb_5000_movies.csv"


# PART 1 - Load data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Task 1: load the dataset with Pandas."""
    return pd.read_csv(path)


# PART 1 - Text preprocessing
def _parse_names(json_str) -> str:
    """Convert TMDB JSON-like columns (genres/keywords) to 'Action Adventure ...'."""
    try:
        items = json.loads(json_str)
        return " ".join(i["name"] for i in items)
    except (TypeError, ValueError, KeyError):
        return ""


def clean_text(text) -> str:
    """Task 2: lowercase -> remove punctuation/special chars -> remove stopwords."""
    if not isinstance(text, str):          # handle missing values
        return ""
    text = text.lower()                                  # 1. lowercase
    text = re.sub(r"[^a-z0-9\s]", " ", text)             # 2. remove punctuation/special chars
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]  # 3. remove stopwords
    return " ".join(words)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2: build the 'clean_text' column."""
    df = df.copy()
    # 4. Handle missing values (replace with empty string)
    for col in ["overview", "genres", "keywords"]:
        if col not in df.columns:
            df[col] = ""
    df["overview"] = df["overview"].fillna("")
    df["genres_text"] = df["genres"].apply(_parse_names)
    df["keywords_text"] = df["keywords"].apply(_parse_names)

    # Combine text features used for recommendations
    combined = df["overview"] + " " + df["genres_text"] + " " + df["keywords_text"]
    df["clean_text"] = combined.apply(clean_text)

    # Drop duplicate titles so item lookup is unambiguous
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    return df


# PART 2 - Vectorization & similarity
def build_tfidf(df: pd.DataFrame, max_features: int = 5000, ngram_range=(1, 2)):
    """Task 3: TF-IDF vectorization."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
    return vectorizer, tfidf_matrix


def compute_similarity(tfidf_matrix) -> np.ndarray:
    """Task 4: cosine similarity between all items (float32 to save memory)."""
    return cosine_similarity(tfidf_matrix).astype(np.float32)


# Model builder (cached in memory)
_MODEL = {}


def get_model():
    """Build the full pipeline once and reuse it."""
    if not _MODEL:
        df = preprocess(load_data())
        _, tfidf_matrix = build_tfidf(df)
        _MODEL["df"] = df
        _MODEL["tfidf"] = tfidf_matrix
        _MODEL["sim"] = compute_similarity(tfidf_matrix)
    return _MODEL["df"], _MODEL["tfidf"], _MODEL["sim"]


# PART 3 - Recommendation function
def recommend(item_name: str, top_n: int = 5) -> pd.DataFrame:
    """Task 5: return top N items most similar to `item_name`."""
    df, _, sim = get_model()

    # 1. Find index of the selected item (case-insensitive)
    matches = df.index[df["title"].str.lower() == item_name.strip().lower()]
    if len(matches) == 0:
        raise ValueError(f"'{item_name}' not found in the dataset.")
    idx = matches[0]

    # 2. Compute similarity scores
    scores = list(enumerate(sim[idx]))

    # 3. Sort (highest first), skip the item itself, return top N
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]

    result = df.loc[[i for i, _ in scores], ["title", "genres_text", "overview"]].copy()
    result["similarity"] = [round(float(s), 4) for _, s in scores]
    return result.reset_index(drop=True)
