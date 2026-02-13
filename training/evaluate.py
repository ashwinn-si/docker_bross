import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load validation data
val = pd.read_csv("data/val.csv")


# Drop Date
if "Date" in val.columns:
    val = val.drop("Date", axis=1)


# Split
X = val.drop("Production", axis=1)
y = val["Production"]


# Load model
model = joblib.load("models/model_v3.pkl")  # change version if needed


# Predict
preds = model.predict(X)


# Metrics
mse = mean_squared_error(y, preds)
rmse = mse ** 0.5

mae = mean_absolute_error(y, preds)
r2 = r2_score(y, preds)


# Average production
avg_prod = y.mean()


# Percentage error
error_pct = (mae / avg_prod) * 100


print("\nEvaluation Results")
print("------------------")
print("RMSE:", round(rmse, 2))
print("MAE :", round(mae, 2))
print("R2  :", round(r2, 3))


print("\nLayman Explanation")
print("------------------")
print(f"On average, the model is wrong by about {round(mae)} units.")
print(f"That is around {round(error_pct,2)}% error compared to normal production.")
print(f"The model understands about {round(r2*100,2)}% of real patterns.")

