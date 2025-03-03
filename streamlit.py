import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

# Load trained LSTM model and scaler
MODEL_PATH = "Final_model.h5"
SCALER_PATH = "scaler.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Define input feature names (same as used in training)
selected_columns = [
    "Z1_ActualPosition", "Z1_CommandPosition", "M1_sequence_number", "X1_OutputCurrent",
    "S1_ActualPosition", "S1_CommandPosition", "Y1_OutputCurrent", "S1_OutputCurrent",
    "clamp_pressure", "feedrate", "M1_CURRENT_FEEDRATE", "S1_CurrentFeedback",
    "X1_DCBusVoltage", "X1_OutputVoltage", "Y1_DCBusVoltage"
]

sequence_length = 10  # Adjust if your model was trained with a different sequence length

# Streamlit UI
st.title("🔧 Tool Wear Prediction (Multi-Output LSTM Model)")
st.markdown("Predict **Tool Condition, Machining Finalization, and Visual Inspection Results**.")

# User inputs
st.sidebar.header("📥 Enter Feature Values")
user_input = {}

for feature in selected_columns:
    user_input[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Convert input into DataFrame
input_df = pd.DataFrame([user_input])

# Preprocess input
input_scaled = scaler.transform(input_df)
input_sequence = np.array([input_scaled] * sequence_length)  # Repeat for LSTM sequence shape
input_sequence = input_sequence.reshape(1, sequence_length, len(selected_columns))  # Reshape for LSTM

# Predict
if st.sidebar.button("Predict"):
    pred_tool_condition, pred_machining_finalized, pred_visual_inspection = model.predict(input_sequence)

    # Convert predictions to readable format
    tool_condition_labels = {0: "Good", 1: "Worn", 2: "Damaged"}
    pred_tool_condition_label = tool_condition_labels[np.argmax(pred_tool_condition[0])]

    machining_finalized_status = "Yes" if pred_machining_finalized[0][0] > 0.5 else "No"
    visual_inspection_status = "Passed" if pred_visual_inspection[0][0] > 0.5 else "Failed"

    # Display results
    st.subheader("🔍 Prediction Results")
    st.write(f"**🛠 Tool Condition:** {pred_tool_condition_label}")
    st.write(f"**🔄 Machining Finalized:** {machining_finalized_status}")
    st.write(f"**👀 Passed Visual Inspection:** {visual_inspection_status}")

    st.success("✅ Prediction Completed!")


