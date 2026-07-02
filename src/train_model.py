"""Train the production model on ALL pooled data and persist it for the app.

Run once (or whenever data changes):  python -m src.train_model
The Streamlit app then just loads models/*.joblib — no retraining at page load.
"""
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.load_data import load_all_sources
from src.preprocess import clean_text

MODEL_DIR = Path("models")


def train_and_save():
    df = load_all_sources()
    print(f"Training on {len(df)} emails from {df['source'].nunique()} sources.")

    # Same config as the validated experiments; fit on the full dataset.
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 1))
    X = vectorizer.fit_transform(df["text"].apply(clean_text))

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, df["label"])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
    print(f"Saved model + vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    train_and_save()
