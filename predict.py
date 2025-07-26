import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

model = joblib.load("models/sklearn_model.joblib")
X, y = fetch_california_housing(return_X_y=True)
preds = model.predict(X)

print(f"R² Score (Predict): {r2_score(y, preds):.4f}")
