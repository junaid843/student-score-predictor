# Student Performance GUI (Streamlit)
# Save this file as student_performance_gui.py
# Place your trained model (joblib / pickle) as `model.pkl` in the same folder if available.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Performance Index Predictor", layout="centered")

st.title("Student Performance Index — GUI")
st.write("Input hours studied and previous scores to predict a performance index.")

MODEL_PATH = "model.pkl"

# Try to load model; if not present, create a simple fallback model
model = None
model_info = ""

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_info = f"Loaded model from {MODEL_PATH}"
    except Exception as e:
        model = None
        model_info = f"Failed to load model ({e}). Using fallback model."
else:
    model_info = "No model.pkl found — using fallback linear model trained on a small sample dataset."

st.caption(model_info)

# Fallback model: simple LinearRegression trained on example data
from sklearn.linear_model import LinearRegression

def make_fallback_model():
    # tiny synthetic dataset
    X = np.array([ [1,50], [2,55], [3,60], [4,65], [5,70], [6,72], [7,78], [8,80], [9,88], [10,90] ])
    # columns: hours_studied, previous_score
    y = np.array([50, 54, 59, 63, 69, 71, 76, 79, 86, 89])
    lr = LinearRegression()
    lr.fit(X, y)
    return lr

if model is None:
    model = make_fallback_model()

# Input widgets for single prediction
st.header("Single prediction")
col1, col2 = st.columns(2)
with col1:
    hours = st.number_input("Hours studied (per week)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
with col2:
    prev_score = st.number_input("Previous score (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

if st.button("Predict Performance Index"):
    X_new = np.array([[hours, prev_score]])
    pred = model.predict(X_new)[0]
    pred = float(pred)
    st.metric(label="Predicted Performance Index", value=f"{pred:.2f}")

    # display model coefficients if available
    try:
        coef = model.coef_
        intercept = model.intercept_
        st.write(f"Model intercept: {intercept:.3f}")
        st.write("Model coefficients (hours_studied, previous_score):", np.round(coef,3))
    except Exception:
        st.write("Model coefficients not available for this model type.")

    # simple diagnostic plot
    fig, ax = plt.subplots()
    ax.scatter([hours], [pred])
    ax.set_xlabel('Hours Studied')
    ax.set_ylabel('Performance Index')
    ax.set_title('Prediction (single point)')
    st.pyplot(fig)

st.markdown("---")

# Batch prediction via CSV upload
st.header("Batch prediction (CSV)")
st.write("Upload a CSV with columns: `hours_studied` and `previous_score`. The app will append `predicted_performance`.")
uploaded = st.file_uploader("Upload CSV file", type=["csv"]) 

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None

    if df is not None:
        st.write("Preview:")
        st.dataframe(df.head())

        # Validate columns (case-insensitive)
        cols = {c.lower(): c for c in df.columns}
        if 'hours_studied' in cols and 'previous_score' in cols:
            X = df[[cols['hours_studied'], cols['previous_score']]].values
            preds = model.predict(X)
            df['predicted_performance'] = preds
            st.success(f"Predicted {len(preds)} rows")
            st.dataframe(df.head())

            # Download button
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime='text/csv')

            # Plot predicted vs previous
            fig2, ax2 = plt.subplots()
            ax2.scatter(df[cols['previous_score']], df['predicted_performance'])
            ax2.set_xlabel('Previous Score')
            ax2.set_ylabel('Predicted Performance')
            ax2.set_title('Previous Score vs Predicted Performance')
            st.pyplot(fig2)

        else:
            st.error("CSV must contain columns named 'hours_studied' and 'previous_score' (case-insensitive).")

st.markdown("---")

# Option to save a toy model from within the app (helpful if user doesn't have model.pkl)
st.header("Model management")
if st.button("Save current model to model.pkl"):
    try:
        joblib.dump(model, MODEL_PATH)
        st.success(f"Saved model to {MODEL_PATH}")
    except Exception as e:
        st.error(f"Failed to save model: {e}")

st.write("\n---\n")
st.caption("Notes: Put your trained scikit-learn model (joblib.dump) in model.pkl in the same folder as this script. For other model types you may need to update load logic.")

# EOF
