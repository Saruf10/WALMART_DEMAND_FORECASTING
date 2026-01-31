# preprocessing.py
import pandas as pd
from pathlib import Path

# -------------------------
# 1️⃣ Define paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure output folder exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 2️⃣ Load datasets
# -------------------------
train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA_DIR / "stores.csv")

# -------------------------
# 3️⃣ Merge datasets
# -------------------------
# Merge train with features
train_merged = pd.merge(
    train, features, on=["Store", "Date", "IsHoliday"], how="left"
)

# Merge with store info
train_merged = pd.merge(train_merged, stores, on="Store", how="left")

# Same for test
test_merged = pd.merge(test, features, on=["Store", "Date", "IsHoliday"], how="left")
test_merged = pd.merge(test_merged, stores, on="Store", how="left")

# -------------------------
# 4️⃣ Handle missing values
# -------------------------
# MarkDown columns: replace NaN with 0 (no promotion)
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
train_merged[markdown_cols] = train_merged[markdown_cols].fillna(0)
test_merged[markdown_cols] = test_merged[markdown_cols].fillna(0)

# CPI & Unemployment: forward-fill then back-fill globally
macro_cols = ["CPI", "Unemployment"]
train_merged[macro_cols] = train_merged[macro_cols].fillna(method="ffill").fillna(method="bfill")
test_merged[macro_cols] = test_merged[macro_cols].fillna(method="ffill").fillna(method="bfill")

# -------------------------
# 5️⃣ Encode categorical variables
# -------------------------
# Store Type → One-hot encoding
train_merged = pd.get_dummies(train_merged, columns=["Type"], drop_first=True)
test_merged = pd.get_dummies(test_merged, columns=["Type"], drop_first=True)

# Convert boolean IsHoliday to int
train_merged["IsHoliday"] = train_merged["IsHoliday"].astype(int)
test_merged["IsHoliday"] = test_merged["IsHoliday"].astype(int)

# -------------------------
# 6️⃣ Create time features
# -------------------------
for df in [train_merged, test_merged]:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day

# -------------------------
# 7️⃣ Save processed datasets
# -------------------------
train_merged.to_csv(DATA_DIR / "train_processed.csv", index=False)
test_merged.to_csv(DATA_DIR / "test_processed.csv", index=False)

print("✅ Preprocessing completed. Processed files saved in 'data/' folder.")
