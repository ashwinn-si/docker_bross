import pandas as pd
import random


# Load drift data
df = pd.read_csv("data/drift_log.csv")


# Fake labels (for now)
# In competition → replace with API call
df["Production"] = df.apply(
    lambda x: random.uniform(200, 300),
    axis=1
)


# Save
df.to_csv("data/labeled_drift.csv", index=False)

print("Saved labeled_drift.csv")
