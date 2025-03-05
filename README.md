

**🔧 Tool Wear Prediction using LSTM & Streamlit**

**📌 Project Overview**

This project aims to predict tool wear conditions in CNC machining using an LSTM (Long Short-Term Memory) model. The goal is to improve manufacturing efficiency by forecasting tool condition, machining finalization, and visual inspection outcomes based on sensor data.


**📂 Dataset Information**

The dataset used for this project contains time-series sensor data from a CNC machine, capturing various operational parameters. The key features include:

Position Data: Z1_ActualPosition, Z1_CommandPosition, S1_ActualPosition, S1_CommandPosition

Electrical Parameters: X1_OutputCurrent, Y1_OutputCurrent, S1_OutputCurrent

Machine Settings: feedrate, clamp_pressure, M1_CURRENT_FEEDRATE

Voltage & Feedback Data: X1_DCBusVoltage, Y1_DCBusVoltage, S1_CurrentFeedback

Target Variables:

  Tool Condition: (Good, Worn, Damaged)
  Machining Finalized: (Yes/No)
  Visual Inspection Result: (Passed/Failed)

    
**🔬 Machine Learning Model**

LSTM Neural Network trained on time-series data
Multi-output prediction: Predicts tool wear conditions, machining status, and inspection results
Data Scaling: MinMaxScaler for preprocessing

**🎨 Web App Features**

The project includes a Streamlit-based interactive web app with:
✅ User input panel to enter sensor values
✅ Real-time predictions powered by the trained LSTM model
✅ Modern UI with a sky blue & black theme
✅ Deployment on AWS EC2

🚀 Installation & Usa
