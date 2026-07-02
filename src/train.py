import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

from scipy.sparse import hstack

from src.load_data import load_dataset, load_spamassassin_ham, load_phishing_pot, load_all_sources
from src.preprocess import clean_text
from src.features import check_ip_url, num_links, suspicious_tld_links

STRUCTURED_NAMES = ["num_ip_urls", "num_links", "suspicious_tld_links"]


def get_split():
    """Load and split once, returning RAW text. Cleaning is left to each
    experiment: the TF-IDF (word) path cleans, the structured (URL) path
    needs the raw markup. Same random_state so results are comparable."""
    df = load_dataset()
    return train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )


def get_diverse_split():
    """Pool ALL sources and split each one 80/20 (stratify by source), so every
    source appears in both train and test. Returns full DataFrames (with a
    'source' column) rather than just text/label, so we can break results down
    by source afterwards."""
    df = load_all_sources()
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["source"], random_state=42
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def evaluate_by_source(model, vectorizer, scaler, test_df, use_structured):
    """Break test-set predictions down per source: recall for phishing sources,
    false-positive rate for legit sources."""
    preds = model.predict(featurize(test_df["text"], vectorizer, use_structured, scaler))
    df = test_df.assign(pred=preds)
    print("\n--- per-source breakdown ---")
    for source, g in df.groupby("source"):
        label = int(g["label"].iloc[0])
        n = len(g)
        pred_phish = int((g["pred"] == 1).sum())
        kind = "phishing, recall" if label == 1 else "legit, FP rate"
        print(f"  {source:14s} ({kind}): {pred_phish / n:.3f}  ({pred_phish}/{n})")


def structured_columns(raw_text):
    """Hand-crafted numeric features, computed from RAW text (URLs intact).
    Returns a dense (n, len(STRUCTURED_NAMES)) array."""
    ip = raw_text.apply(check_ip_url).values.reshape(-1, 1)
    links = raw_text.apply(num_links).values.reshape(-1, 1)
    sus = raw_text.apply(suspicious_tld_links).values.reshape(-1, 1)
    return np.hstack([ip, links, sus])


def featurize(raw_text, vectorizer, use_structured, scaler=None):
    """Turn raw email text into the model's feature matrix, the SAME way for
    train, test, and OOD data. TF-IDF runs on cleaned text; structured features
    run on raw text, get scaled (so their raw magnitude doesn't dominate the
    ~0-1 TF-IDF values), and are stacked on the side."""
    tfidf = vectorizer.transform(raw_text.apply(clean_text))
    if use_structured:
        cols = scaler.transform(structured_columns(raw_text))
        return hstack([tfidf, cols])
    return tfidf


def evaluate(X_train_vec, X_test_vec, y_train, y_test, label=""):
    """Train a LogReg on the given feature matrices and report metrics."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    phish_recall = recall_score(y_test, y_pred, pos_label=1)

    print(f"\n=== {label} (in-distribution test) ===")
    print("Accuracy:", acc)
    print("Phishing recall:", phish_recall)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, (acc, phish_recall)


def show_top_features(model, feature_names, n=20):
    """Print the features pushing hardest toward each class."""
    feats = np.array(feature_names)
    coefs = model.coef_[0]
    print("Top phishing signals:", feats[np.argsort(coefs)[-n:]])
    print("Top legit signals:   ", feats[np.argsort(coefs)[:n]])


def run_model(X_train_text, X_test_text, y_train, y_test, use_structured, label):
    """Fit TF-IDF (optionally + structured features) and evaluate in-distribution.
    Returns the fitted model + vectorizer so we can reuse them on OOD data."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 1))
    vectorizer.fit(X_train_text.apply(clean_text))          # learn vocab on train only

    # Fit the scaler on TRAIN structured columns only (same leakage rule as TF-IDF).
    scaler = StandardScaler().fit(structured_columns(X_train_text)) if use_structured else None

    X_train_vec = featurize(X_train_text, vectorizer, use_structured, scaler)
    X_test_vec = featurize(X_test_text, vectorizer, use_structured, scaler)

    model, scores = evaluate(X_train_vec, X_test_vec, y_train, y_test, label=label)

    names = list(vectorizer.get_feature_names_out())
    if use_structured:
        names += STRUCTURED_NAMES                            # keep names aligned with coefs
    show_top_features(model, names)
    return model, vectorizer, scaler, scores


def evaluate_ood(model, vectorizer, scaler, ood_records, name, use_structured):
    """Push an unseen, single-label corpus through the fitted model (transform,
    never fit). Reports recall if the set is all phishing, FP rate if all legit."""
    df = pd.DataFrame(ood_records)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)  # drop empty bodies, as training does

    preds = model.predict(featurize(df["text"], vectorizer, use_structured, scaler))

    true_label = int(df["label"].iloc[0])
    n = len(preds)
    pred_phish = int((preds == 1).sum())

    print(f"\n--- {name} ---")
    print(f"emails: {n} (all label={true_label})")
    if true_label == 1:
        print(f"caught as phishing: {pred_phish}  ->  recall: {pred_phish / n:.3f}")
    else:
        print(f"wrongly flagged as phishing: {pred_phish}  ->  false-positive rate: {pred_phish / n:.3f}")


def experiment_ood():
    """Experiment 1: train on Nazario + Enron only, test on FULLY UNSEEN sources.
    Exposes the source confound and the structured-feature precision/recall
    tradeoff (OOD phishing recall ~0.58, OOD legit FP explodes with features)."""
    print("\n########## Experiment 1: narrow training -> out-of-distribution test ##########")
    X_train_text, X_test_text, y_train, y_test = get_split()
    ham = load_spamassassin_ham()
    pot = load_phishing_pot()
    for use_structured, label in [(False, "TF-IDF baseline"), (True, "TF-IDF + structured")]:
        model, vectorizer, scaler, _ = run_model(
            X_train_text, X_test_text, y_train, y_test, use_structured, label
        )
        evaluate_ood(model, vectorizer, scaler, ham, "OOD legit — SpamAssassin ham", use_structured)
        evaluate_ood(model, vectorizer, scaler, pot, "OOD phishing — phishing_pot", use_structured)


def experiment_diverse():
    """Experiment 2: pool ALL sources, split each into train + test. Shows that
    diverse training data fixes the generalization gap at the root — phishing_pot
    recall jumps 0.58 -> ~0.99 with no precision cost."""
    print("\n########## Experiment 2: diverse training -> per-source test ##########")
    train_df, test_df = get_diverse_split()
    print("train sources:", dict(train_df["source"].value_counts()))
    print("test sources: ", dict(test_df["source"].value_counts()))
    model, vectorizer, scaler, _ = run_model(
        train_df["text"], test_df["text"], train_df["label"], test_df["label"],
        use_structured=False, label="TF-IDF, all sources",
    )
    evaluate_by_source(model, vectorizer, scaler, test_df, use_structured=False)


if __name__ == "__main__":
    experiment_ood()
    experiment_diverse()
