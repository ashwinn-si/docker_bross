from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

from drift import detect_drift, save_drift
from preprocess import preprocess


app = Flask(__name__)


# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load model
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_v3.pkl")
model = joblib.load(MODEL_PATH)


# Load train data for drift stats
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
train_data = pd.read_csv(TRAIN_PATH)


num_cols = ["Day_of_Year", "Start_Hour", "End_Hour"]


train_stats = {
    col: train_data[col].mean()
    for col in num_cols
}


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # If single object is sent, convert to list
    if isinstance(data, dict):
        data = [data]

    # Convert list of JSON → DataFrame
    df = preprocess(data)

    # Predict for all rows
    preds = model.predict(df)

    results = []

    for i, pred in enumerate(preds):

        row_df = df.iloc[[i]]

        # Drift check per row
        if detect_drift(train_stats, row_df[num_cols]):
            save_drift(row_df)

        results.append(float(pred))

    return jsonify({
        "predictions": results
    })




@app.route("/health")
def health():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
