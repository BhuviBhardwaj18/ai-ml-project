import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

X_train = np.array([[0], [1], [2], [3], [4], [5]])
y_train = np.array([0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")
print("Model saved successfully")
