import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()

df = pd.DataFrame(
    data.data,
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

df["target"] = data.target

df.to_csv("data/iris.csv", index=False)

print("Data preprocessing completed")
print("Dataset shape:", df.shape)
