import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("models/model.pkl")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

metrics = {
    "accuracy": round(accuracy, 4)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Evaluation completed")
print("Accuracy:", accuracy)
