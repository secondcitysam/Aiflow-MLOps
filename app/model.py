import joblib

model = joblib.load("models/model.pkl")


def predict(features):
    prediction = model.predict([features])
    return int(prediction[0])
