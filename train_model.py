
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

data = pd.read_csv("dataset.csv")

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_job_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model trained successfully!")
