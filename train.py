from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib, os

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

score = r2_score(y_test, model.predict(X_test))
print(f"R² Score: {score:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sklearn_model.joblib")
