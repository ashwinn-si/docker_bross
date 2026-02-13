import os


# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Drift file path
DRIFT_PATH = os.path.join(BASE_DIR, "data", "drift_log.csv")


def detect_drift(train_stats, new_df, threshold=0.25):

    drift = False

    for col in train_stats:

        new_mean = new_df[col].mean()

        diff = abs(train_stats[col] - new_mean)

        if diff > threshold:
            drift = True

    return drift


def save_drift(df):

    # Create file if not exists
    if not os.path.exists(DRIFT_PATH):
        df.to_csv(DRIFT_PATH, index=False)
    else:
        df.to_csv(DRIFT_PATH, mode="a", header=False, index=False)
