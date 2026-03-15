import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Load model and data
model = joblib.load("f1_model.pkl")
encoder = joblib.load("encoder.pkl")
df = joblib.load("data.pkl")

st.title("Grid to Flag: F1 Race Predictor")

# Inputs
circuit = st.selectbox("Select Circuit", sorted(df["circuit"].unique()))
grid = st.slider("Starting Grid Position", 1, 20, 5)

if st.button("Predict Finish"):

    circuit_encoded = encoder.transform([circuit])[0]

    prediction = model.predict([[grid, circuit_encoded]])[0]

    st.subheader("Prediction")

    lower = max(1, int(prediction) - 1)
    upper = min(20, int(prediction) + 1)

    st.write(f"Predicted Finish Range: **P{lower} - P{upper}**")

    # Chart
    circuit_data = df[df["circuit"] == circuit]

    chart = circuit_data.groupby("grid_position")["finish_position"].mean().reset_index()

    fig = px.bar(
        chart,
        x="grid_position",
        y="finish_position",
        labels={"grid_position": "Grid Position", "finish_position": "Average Finish"},
        title=f"Average Finish by Grid Position at {circuit}"
    )

    st.plotly_chart(fig)

    st.write(
        f"Historically at **{circuit}**, drivers starting around P{grid} tend to finish around position **{prediction:.2f}**."
    )
