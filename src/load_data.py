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

if __name__ == "__main__":
    df = load_dataset()
    print(df.shape)
    print(df["label"].value_counts())
    print(df.sample(5))