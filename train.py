import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("intent-classification")

df = pd.read_csv("dataset.csv")

vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5))),
])

model = Pipeline([
    ("features", vectorizer),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

with mlflow.start_run():
    model.fit(df["text"], df["intent"])
    preds = model.predict(df["text"])
    acc = accuracy_score(df["intent"], preds)

    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("word_ngram_range", "(1, 2)")
    mlflow.log_param("char_ngram_range", "(3, 5)")
    mlflow.log_metric("train_accuracy", acc)
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="intent-classifier",
        pip_requirements=["scikit-learn", "pandas", "joblib"],
    )

joblib.dump(model, "intent_model.pkl")
print("Model trained, registered in MLflow SQLite, and saved as intent_model.pkl")
