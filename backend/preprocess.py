import pandas as pd


def preprocess(data):

    df = pd.DataFrame([data])

    # Drop Date if present
    if "Date" in df.columns:
        df = df.drop("Date", axis=1)

    return df
