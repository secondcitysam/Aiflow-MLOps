import joblib
import pandas as pd


def test_model_exists():
    model = joblib.load("models/model.pkl")
    assert model is not None


def test_model_prediction():
    model = joblib.load("models/model.pkl")

    features = pd.DataFrame([
        [5.1, 3.5, 1.4, 0.2]
    ], columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ])

    prediction = model.predict(features)

    assert prediction[0] == 0

#df


