import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Kaggle dataset expected columns:
# id, title, author, text, label

df = pd.read_csv("train.csv")

df = df.fillna("")
df["content"] = df["title"] + " " + df["author"] + " " + df["text"]

X = df["content"]
y = df["label"]  # 0 = fake, 1 = real (depends on dataset source)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
    ("classifier", PassiveAggressiveClassifier(max_iter=50))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Model Accuracy: {acc:.2f}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl saved successfully.")
