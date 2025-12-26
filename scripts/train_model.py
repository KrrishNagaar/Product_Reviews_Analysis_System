import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scripts.preprocess import clean_text

DATA_PATH = "data/reviews.csv"
MAX_FEATURES = 8000
N_SPLITS = 5
HYPER_ALPHAS = [1e-4, 1e-5, 5e-5]

# Load & shuffle
df = pd.read_csv(DATA_PATH)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["review_text"] = df["review_text"].astype(str)
df["cleaned_review"] = df["review_text"].apply(clean_text)

y = df["label"].map({"real": 1, "fake": 0}).values
X_text = df["cleaned_review"].values

# TF-IDF sparse
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1,3), sublinear_tf=True)
X_all = vectorizer.fit_transform(X_text)
print("TF-IDF shape:", X_all.shape)

# Quick hold-out for tuning
val_split = int(X_all.shape[0] * 0.8)

best_alpha = None
best_auc = -1.0
for a in HYPER_ALPHAS:
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=a, max_iter=1000)
    model.fit(X_all[:val_split], y[:val_split])
    probs = model.predict_proba(X_all[val_split:])[:,1]
    fpr, tpr, _ = roc_curve(y[val_split:], probs)
    score = auc(fpr, tpr)
    print(f"alpha={a} -> AUC={score:.4f}")
    if score > best_auc:
        best_auc = score
        best_alpha = a

print("Best alpha:", best_alpha)

# Train final model on all data
final_model = SGDClassifier(loss="log_loss", penalty="l2", alpha=best_alpha, max_iter=1000)
final_model.fit(X_all, y)

os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/review_model_final.pkl")
joblib.dump(vectorizer, "models/vectorizer_final.pkl")
print("Saved models/review_model_final.pkl and models/vectorizer_final.pkl")
