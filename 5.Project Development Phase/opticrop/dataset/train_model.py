import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("Crop_recommendation.csv")
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "crop_model.pkl")

print("Model trained successfully!")