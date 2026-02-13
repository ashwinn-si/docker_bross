import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw.csv")

X = df.drop("Production", axis=1)
y = df["Production"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train["Production"] = y_train
X_val["Production"] = y_val

X_train.to_csv("data/train.csv", index=False)
X_val.to_csv("data/val.csv", index=False)

print("Split done")
