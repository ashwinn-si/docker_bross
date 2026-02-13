import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor


# Load old + new data
old = pd.read_csv("data/train.csv")
new = pd.read_csv("data/labeled_drift.csv")


# Merge
df = pd.concat([old, new])


# Drop Date
if "Date" in df.columns:
    df = df.drop("Date", axis=1)


# Split
X = df.drop("Production", axis=1)
y = df["Production"]


num_cols = ["Day_of_Year", "Start_Hour", "End_Hour"]

cat_cols = ["Day_Name", "Month_Name", "Season", "Source"]


# Preprocessor
prep = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])


# Improved model
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    random_state=42
)


pipe = Pipeline([
    ("prep", prep),
    ("model", model)
])


# Train
pipe.fit(X, y)


# Save new model
joblib.dump(pipe, "models/model_v3.pkl")

print("Saved model_v3.pkl")
