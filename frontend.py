import streamlit as st
import requests

st.set_page_config(layout="centered")
st.title("🏥 Sepsis Early Warning System")
st.write("This tool uses an XGBoost model to predict sepsis risk based on key patient vitals. Your model was trained on 39 features; any not entered here will be filled with their average value.")

# Based on your feature importance plot, HR, SBP, Temp, O2Sat, and WBC are key.
st.header("Enter Key Patient Vitals")

col1, col2 = st.columns(2)

with col1:
    hr = st.slider("Heart Rate (HR)", 40, 200, 140, help="High heart rate can be a sign of infection.")
    sbp = st.slider("Systolic Blood Pressure (SBP)", 50, 200, 85, help="Low blood pressure is a critical indicator of sepsis.")
    temp = st.slider("Temperature (°C)", 35.0, 42.0, 39.5, help="Fever is a common symptom of sepsis.")

with col2:
    o2sat = st.slider("Oxygen Saturation (O2Sat %)", 70, 100, 85, help="Low oxygen levels can indicate organ stress.")
    wbc = st.slider("White Blood Cell Count (x10^3/µL)", 1.0, 50.0, 18.0, help="A high WBC count suggests the body is fighting an infection.")

# You found threshold 0.2-0.3 gave high recall. Let's use 0.3 for the demo.
OPTIMAL_THRESHOLD = 0.3

if st.button("Analyze Sepsis Risk", type="primary"):
    # The data sent to the backend only includes what the user entered
    api_data = {
        "HR": hr,
        "O2Sat": o2sat,
        "Temp": temp,
        "SBP": sbp,
        "WBC": wbc
    }

    try:
        # Send request to the backend
        response = requests.post("https://sepsis-detector.onrender.com/predict", json=api_data)
        result = response.json()

        if "probability" in result:
            prob = result['probability']
            is_sepsis = prob >= OPTIMAL_THRESHOLD

            st.subheader("Prediction Result")
            if is_sepsis:
                st.error(f"High Risk of Sepsis Detected")
                st.info(f"The prediction is 'High Risk' because the risk score is above the optimal threshold of {OPTIMAL_THRESHOLD*100}%.")
            else:
                st.success(f"Low Risk of Sepsis")
                st.info(f"The prediction is 'Low Risk' because the risk score is below the optimal threshold of {OPTIMAL_THRESHOLD*100}%.")
            st.metric(label="Sepsis Risk Score", value=f"{prob*100:.1f}%")
            st.progress(prob)

        else:
            st.error(f"Error from API: {result.get('error', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend. Is the 'app.py' server running?")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
