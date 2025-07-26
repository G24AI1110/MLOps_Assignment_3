import joblib
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

X, y = fetch_california_housing(return_X_y=True)
X_tensor = torch.tensor(X, dtype=torch.float32)

sk_model = joblib.load("models/sklearn_model.joblib")
weights = sk_model.coef_.astype(np.float32)
bias = sk_model.intercept_.astype(np.float32)

y_pred_sk = sk_model.predict(X)
r2_sk = r2_score(y, y_pred_sk)
print(f" R² Score (Sklearn Model): {r2_sk:.4f}")

unquant_params = {"weights": weights, "bias": bias}
joblib.dump(unquant_params, "models/unquant_params.joblib")

max_w = np.max(np.abs(weights))
scale_w = 127.0 / max_w
quant_weights = np.round(weights * scale_w).astype(np.int8)

max_b = np.max(np.abs(bias))
scale_b = 127.0 / max_b if max_b != 0 else 1.0
quant_bias = np.round(bias * scale_b).astype(np.int8)

quant_params = {
    "weights": quant_weights,
    "bias": quant_bias,
    "scale_w": scale_w,
    "scale_b": scale_b
}
joblib.dump(quant_params, "models/quant_params.joblib")


class SimpleNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

model_torch = SimpleNN(X.shape[1])


dequant_weights = quant_weights.astype(np.float32) / scale_w
dequant_bias = quant_bias.astype(np.float32) / scale_b

model_torch.linear.weight.data = torch.tensor(dequant_weights.reshape(1, -1), dtype=torch.float32)
model_torch.linear.bias.data = torch.tensor([dequant_bias.item()], dtype=torch.float32)


with torch.no_grad():
    y_pred_q = model_torch(X_tensor).squeeze().numpy()
    r2_q = r2_score(y, y_pred_q)

print(f" R² Score (Quantized Model): {r2_q:.4f}")

uq_size = os.path.getsize("models/unquant_params.joblib") / 1024
q_size = os.path.getsize("models/quant_params.joblib") / 1024

print(f" Unquantized Param Size: {uq_size:.2f} KB")
print(f" Quantized Param Size: {q_size:.2f} KB")
