import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# === Load data
X, y = fetch_california_housing(return_X_y=True)

# === Load model
model = joblib.load("models/sklearn_model.joblib")
weights = model.coef_.astype(np.float32)
bias = np.array([model.intercept_], dtype=np.float32)

# === Evaluate original model
y_pred = model.predict(X)
r2_orig = r2_score(y, y_pred)
print(f"✅ R² Score (Sklearn Model): {r2_orig:.4f}")

# === Save unquantized model
np.save("models/unquant_weights.npy", weights)
np.save("models/unquant_bias.npy", bias)

# === Fake quantization: convert to float16
weights_q = weights.astype(np.float16)
bias_q = bias.astype(np.float16)

np.save("models/quant_weights.npy", weights_q)
np.save("models/quant_bias.npy", bias_q)

# === Evaluate with float16 weights
y_pred_q = X @ weights_q + bias_q
r2_q = r2_score(y, y_pred_q)
print(f"✅ R² Score (Quantized Model): {r2_q:.4f}")

# === Compare sizes
uq_size = (
    os.path.getsize("models/unquant_weights.npy") +
    os.path.getsize("models/unquant_bias.npy")
) / 1024

q_size = (
    os.path.getsize("models/quant_weights.npy") +
    os.path.getsize("models/quant_bias.npy")
) / 1024

print(f"📦 Unquantized Param Size: {uq_size:.2f} KB")
print(f"📦 Quantized Param Size: {q_size:.2f} KB")
