from src.load_data import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.preprocess import clean_text

df = load_dataset()
df["text"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], 
                                                    test_size=0.2, 
                                                    stratify=df["label"],
                                                    random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 1))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import numpy as np
feats = np.array(vectorizer.get_feature_names_out())
coefs = model.coef_[0]
print("Top phishing signals:", feats[np.argsort(coefs)[-20:]])
print("Top legit signals:   ", feats[np.argsort(coefs)[:20]])
