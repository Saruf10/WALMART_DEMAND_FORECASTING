# real_time_simulation.py
import pandas as pd
import joblib
from pathlib import Path
import time

# -------------------------
# 1️⃣ Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# -------------------------
# 2️⃣ Load preprocessed test data
# -------------------------
test = pd.read_csv(DATA_DIR / "test_processed.csv", parse_dates=["Date"])

# -------------------------
# 3️⃣ Ensure correct dtypes for LightGBM
# -------------------------

# Numeric columns
numeric_cols = [
    "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3",
    "MarkDown4", "MarkDown5", "CPI", "Unemployment", "Size",
    "Year", "Month", "Week", "Day"
]
for col in numeric_cols:
    if col in test.columns:
        test[col] = pd.to_numeric(test[col], errors="coerce")

# Boolean columns (including one-hot encoded types)
bool_cols = ["IsHoliday", "Type_B", "Type_C"]
for col in bool_cols:
    if col in test.columns:
        test[col] = test[col].astype(bool)

# Fill remaining NaNs with 0
test.fillna(0, inplace=True)

# -------------------------
# 4️⃣ Load trained LightGBM model
# -------------------------
model = joblib.load(DATA_DIR / "lgb_model.pkl")

# -------------------------
# 5️⃣ Real-time simulation
# -------------------------
print("🚀 Starting real-time simulation...")

# Sort chronologically
test_sorted = test.sort_values("Date")

predictions = []

for idx, row in test_sorted.iterrows():
    # Drop non-feature columns for prediction
    X_new = row.drop(["Store", "Dept", "Date"]).to_frame().T

    # Ensure all dtypes match training
    X_new = X_new.astype(float, errors="ignore")
    
    pred = model.predict(X_new)[0]

    # Print live prediction
    print(f"Date: {row['Date'].date()}, Store: {row['Store']}, Dept: {row['Dept']}, Predicted Weekly Sales: {pred:.2f}")

    predictions.append({
        "Date": row["Date"],
        "Store": row["Store"],
        "Dept": row["Dept"],
        "Predicted_Weekly_Sales": pred
    })

    # Optional delay to simulate real-time streaming
    # time.sleep(0.01)

# -------------------------
# 6️⃣ Save all predictions
# -------------------------
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(DATA_DIR / "real_time_predictions.csv", index=False)

print("✅ Real-time simulation completed. Predictions saved to 'data/real_time_predictions.csv'")
