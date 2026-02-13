from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import logger
import logging

from drift import detect_drift, save_drift
from preprocess import preprocess


app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

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

    logger.info(f"Received data: {data}")

    # Must be list
    if not isinstance(data, list):
        return jsonify({"error": "Input must be a list of objects"}), 400

    results = []

    for item in data:

        # Preprocess ONE record
        df = preprocess(item)

  

        if df is None or df.empty:
            return jsonify({"error": "Invalid input data"}), 400

        # Predict ONE record
        pred = model.predict(df)[0]

        # Drift check
        if detect_drift(train_stats, df[num_cols]):
            save_drift(df)

        results.append(float(pred))

    return jsonify({
        "predictions": results
    })




@app.route("/health")
def health():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
