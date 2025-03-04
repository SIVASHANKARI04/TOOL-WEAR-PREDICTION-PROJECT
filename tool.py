import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
from fpdf import FPDF

# Load trained LSTM model and scaler
MODEL_PATH = "Final_model.h5"
SCALER_PATH = "scaler.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

selected_columns = [
    "Z1_ActualPosition", "Z1_CommandPosition", "M1_sequence_number", "X1_OutputCurrent",
    "S1_ActualPosition", "S1_CommandPosition", "Y1_OutputCurrent", "S1_OutputCurrent",
    "clamp_pressure", "feedrate", "M1_CURRENT_FEEDRATE", "S1_CurrentFeedback",
    "X1_DCBusVoltage", "X1_OutputVoltage", "Y1_DCBusVoltage"
]

sequence_length = 10

# Streamlit Page Navigation
st.set_page_config(page_title="Tool Wear Prediction", page_icon="🔧", layout="wide")

menu = ["Home", "Prediction"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.title("🔬 Tool Wear Prediction using LSTM")
    st.markdown(
        """
        ## 🧠 Neural Network Architecture
        - **Model Type:** LSTM (Long Short-Term Memory)
        - **Layers:** Input → LSTM Layers → Dense (Output)
        - **Optimizer:** Adam
        - **Activation Functions:** ReLU, Softmax

        ## 🛠 How Prediction Works?
        - Enter real-time machining parameters.
        - Model predicts **Tool Condition, Machining Finalization, and Visual Inspection**.
        - Download results as a **PDF**.
        """
    )

elif choice == "Prediction":
    st.title("🎨 Tool Wear Prediction")
    st.markdown("Fill in the details below and predict tool wear conditions.")
    
    st.sidebar.header("📥 Enter Feature Values")
    user_input = {feature: st.sidebar.number_input(f"{feature}", value=0.0) for feature in selected_columns}

    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    input_sequence = np.array([input_scaled] * sequence_length)
    input_sequence = input_sequence.reshape(1, sequence_length, len(selected_columns))

    if st.sidebar.button("Predict"):
        pred_tool_condition, pred_machining_finalized, pred_visual_inspection = model.predict(input_sequence)

        tool_condition_labels = {0: "Good", 1: "Worn", 2: "Damaged"}
        pred_tool_condition_label = tool_condition_labels[np.argmax(pred_tool_condition[0])]
        machining_finalized_status = "Yes" if pred_machining_finalized[0][0] > 0.5 else "No"
        visual_inspection_status = "Passed" if pred_visual_inspection[0][0] > 0.5 else "Failed"

        st.subheader("🔍 Prediction Results")
        st.write(f"**🛠 Tool Condition:** {pred_tool_condition_label}")
        st.write(f"**🔄 Machining Finalized:** {machining_finalized_status}")
        st.write(f"**👀 Passed Visual Inspection:** {visual_inspection_status}")
        st.success("✅ Prediction Completed!")

        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Tool Wear Prediction Report", ln=True, align='C')
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Tool Condition: {pred_tool_condition_label}", ln=True)
            pdf.cell(200, 10, txt=f"Machining Finalized: {machining_finalized_status}", ln=True)
            pdf.cell(200, 10, txt=f"Visual Inspection: {visual_inspection_status}", ln=True)
            pdf.output("Tool_Wear_Report.pdf")
            return "Tool_Wear_Report.pdf"

        if st.button("📄 Download PDF Report"):
            pdf_file = create_pdf()
            with open(pdf_file, "rb") as f:
                st.download_button(label="Download Report", data=f, file_name="Tool_Wear_Report.pdf", mime="application/pdf")

