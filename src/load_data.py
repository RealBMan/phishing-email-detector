import mailbox
import pandas as pd
from pathlib import Path
from src.parse import extract_body_from_message, extract_body_from_string

def load_nazario_data():
    records = []
    for mbox_path in Path("data/nazario").glob("*"):
        mbox = mailbox.mbox(str(mbox_path))
        for msg in mbox:
            body = extract_body_from_message(msg)
            records.append({"text": body, "label": 1})
    return records

def load_enron_data():
    records = []
    for ham_dir in Path('data').glob('enron*/ham'):
        for file in ham_dir.glob('*.txt'):
            text = file.read_text(encoding='latin-1')
            records.append({'text': text, 'label': 0})
    return records

def load_dataset():
    records = []
    records.extend(load_nazario_data())
    records.extend(load_enron_data())
    df = pd.DataFrame(records)
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)
    return df

def load_spamassassin_ham():
    """Out-of-distribution LEGIT emails (label 0) from a non-Enron source.
    Used only as a held-out test set, never for training."""
    records = []
    for file in Path('data/easy_ham').glob('[0-9]*'):
        raw = file.read_text(encoding='latin-1')
        body = extract_body_from_string(raw)
        records.append({'text': body, 'label': 0})
    return records

def load_all_sources():
    """Combine every source into one DataFrame tagged with its origin, so we can
    train on a diverse mix and split each source into both train and test."""
    sources = [
        (load_enron_data(), "enron"),            # legit
        (load_nazario_data(), "nazario"),        # phishing
        (load_spamassassin_ham(), "spamassassin"),  # legit
        (load_phishing_pot(), "phishing_pot"),   # phishing
    ]
    frames = []
    for records, name in sources:
        df = pd.DataFrame(records)
        df["source"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df

def load_phishing_pot():
    """Out-of-distribution PHISHING emails (label 1) from a different source
    than Nazario (phishing_pot .eml files). Held-out test set, never trained on."""
    records = []
    for file in Path('data/phishing-email').glob('*.eml'):
        raw = file.read_text(encoding='latin-1')
        body = extract_body_from_string(raw)
        records.append({'text': body, 'label': 1})
    return records

if __name__ == "__main__":
    df = load_dataset()
    print(df.shape)
    print(df["label"].value_counts())
    print(df.sample(5))