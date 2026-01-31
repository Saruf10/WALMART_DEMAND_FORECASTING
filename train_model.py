# train_model.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import joblib

# -------------------------
# 1️⃣ Define paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# -------------------------
# 2️⃣ Load preprocessed data
# -------------------------
train = pd.read_csv(DATA_DIR / "train_processed.csv", parse_dates=["Date"])
test = pd.read_csv(DATA_DIR / "test_processed.csv", parse_dates=["Date"])

# -------------------------
# 3️⃣ Feature selection
# -------------------------
drop_cols = ["Weekly_Sales", "Date", "Store", "Dept"]
X = train.drop(columns=drop_cols)
y = train["Weekly_Sales"]
X_test = test.drop(columns=["Date", "Store", "Dept"])

# -------------------------
# 4️⃣ Train-validation split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 5️⃣ Initialize LightGBM Regressor
# -------------------------
model = LGBMRegressor(
    objective="regression",
    learning_rate=0.1,
    num_leaves=31,
    n_estimators=1000,
    random_state=42
)

# -------------------------
# 6️⃣ Train model with early stopping callback
# -------------------------
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# -------------------------
# 7️⃣ Evaluate on validation set
# -------------------------
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"✅ Validation RMSE: {rmse:.2f}")

# -------------------------
# 8️⃣ Predict on test set
# -------------------------
test_preds = model.predict(X_test)
test["Weekly_Sales_Predicted"] = test_preds
test[["Store", "Dept", "Date", "Weekly_Sales_Predicted"]].to_csv(
    DATA_DIR / "test_predictions.csv", index=False
)
print("✅ Predictions saved to 'data/test_predictions.csv'")

# -------------------------
# 9️⃣ Feature importance plot
# -------------------------
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(12,6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig(DATA_DIR / "feature_importance.png")
plt.show()
print("✅ Feature importance plot saved to 'data/feature_importance.png'")
# Save trained model for real-time simulation or deployment
joblib.dump(model, DATA_DIR / "lgb_model.pkl")
print("✅ Trained model saved to 'data/lgb_model.pkl'")