import pandas as pd
import matplotlib.pyplot as plt
import os


# -----------------------------
# Create plots folder
# -----------------------------
PLOT_DIR = "data/drift_plots"

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


# -----------------------------
# Load data
# -----------------------------
original = pd.read_csv("data/train.csv")
drifted = pd.read_csv("data/drift_log.csv")


# -----------------------------
# Record counts
# -----------------------------
total_original = len(original)
total_drifted = len(drifted)

drift_percent = (total_drifted / total_original) * 100


# -----------------------------
# Drop Date column
# -----------------------------
if "Date" in original.columns:
    original = original.drop("Date", axis=1)

if "Date" in drifted.columns:
    drifted = drifted.drop("Date", axis=1)


# -----------------------------
# Numerical columns
# -----------------------------
num_cols = ["Day_of_Year", "Start_Hour", "End_Hour"]


# -----------------------------
# Force numeric conversion
# -----------------------------
for col in num_cols:
    original[col] = pd.to_numeric(original[col], errors="coerce")
    drifted[col] = pd.to_numeric(drifted[col], errors="coerce")




# -----------------------------
# Analysis
# -----------------------------
results = []


for col in num_cols:

    orig_mean = original[col].mean()
    drift_mean = drifted[col].mean()

    mean_diff = drift_mean - orig_mean
    pct_change = (mean_diff / orig_mean) * 100

    drift_flag = "YES" if abs(pct_change) > 10 else "NO"


    results.append([
        col,
        round(orig_mean, 2),
        round(drift_mean, 2),
        round(pct_change, 2),
        drift_flag
    ])


    # -----------------------------
    # Plotting
    # -----------------------------
    plt.figure()

    plt.hist(
        original[col],
        bins=30,
        alpha=0.6,
        label="Original"
    )

    plt.hist(
        drifted[col],
        bins=30,
        alpha=0.6,
        label="Drifted"
    )

    plt.title(f"Drift Analysis: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.legend()

    file_path = os.path.join(PLOT_DIR, f"{col}_drift.png")

    plt.savefig(file_path)
    plt.close()

    print(f"Saved plot: {file_path}")


# -----------------------------
# Print table
# -----------------------------
print(f"\n{'Feature':<15}{'Orig Mean':<12}{'Drift Mean':<12}{'% Change':<12}{'Drift'}")
print("-" * 55)

for r in results:
    print(f"{r[0]:<15}{r[1]:<12}{r[2]:<12}{r[3]:<12}{r[4]}")


# -----------------------------
# Save CSV report
# -----------------------------
report = pd.DataFrame(
    results,
    columns=[
        "Feature",
        "Original_Mean",
        "Drift_Mean",
        "Percent_Change",
        "Drift"
    ]
)

report_path = "data/drift_report.csv"

report.to_csv(report_path, index=False)

print(f"\nSaved report: {report_path}")
