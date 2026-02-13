import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# Load data
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")


# Drop Date
train = train.drop("Date", axis=1)
val = val.drop("Date", axis=1)


# Split X, y
X_train = train.drop("Production", axis=1)
y_train = train["Production"]

X_val = val.drop("Production", axis=1)
y_val = val["Production"]


# Columns
num_cols = [
    "Day_of_Year",
    "Start_Hour",
    "End_Hour"
]

cat_cols = [
    "Day_Name",
    "Month_Name",
    "Season",
    "Source"
]


# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)


# Model
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)


# Pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])


# Train
pipeline.fit(X_train, y_train)


# Validate
preds = pipeline.predict(X_val)

mse = mean_squared_error(y_val, preds)
rmse = mse ** 0.5


print("RMSE:", rmse)


# Save
joblib.dump(pipeline, "models/model_v1.pkl")

print("Saved model_v1.pkl")
