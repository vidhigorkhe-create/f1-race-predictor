import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load datasets
circuits = pd.read_csv("data/circuits.csv")
races = pd.read_csv("data/races.csv")
results = pd.read_csv("data/results.csv")

# Select useful columns
circuits = circuits[["circuitId", "name"]]
races = races[["raceId", "circuitId"]]
results = results[["raceId", "grid", "positionOrder"]]

# Merge datasets
df = results.merge(races, on="raceId")
df = df.merge(circuits, on="circuitId")

# Rename columns
df = df.rename(columns={
    "grid": "grid_position",
    "positionOrder": "finish_position",
    "name": "circuit"
})

# Clean data
df = df[df["grid_position"] > 0]
df = df[df["finish_position"] > 0]

# Encode circuit
encoder = LabelEncoder()
df["circuit_encoded"] = encoder.fit_transform(df["circuit"])

# Features and target
X = df[["grid_position", "circuit_encoded"]]
y = df["finish_position"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and encoder
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/f1_model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(df, "model/data.pkl")

print("Model trained and saved successfully!")