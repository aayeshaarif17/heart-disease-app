"""Heart Disease Prediction System - Streamlit app (AIC354, Assignment 2, Task 2).

Uses the SAME trained SVC model and the SAME label encoders produced by
heart_disease_prediction.ipynb, with exactly four inputs:
Sex, ChestPain, ExerciseAngina, Thal.
"""
import os
import pickle

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_svc_trained_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "heart_label_encoders.pkl")
INPUT_COLUMNS = ["Sex", "ChestPain", "ExerciseAngina", "Thal"]  # same order as training

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


st.title("❤️ Heart Disease Prediction System")
st.write("This app uses a Support Vector Classifier trained on the UCI Cleveland Heart Disease dataset. Select four patient details and click Predict to see whether heart disease is likely.")

try:
    model, encoders = load_artifacts()
except Exception as e:  # missing / incompatible model files
    st.error(f"Could not load the trained model or encoders: {e}")
    st.stop()

# Options come straight from the saved encoders, so they can never differ from training.
sex = st.selectbox("Sex", list(encoders["Sex"].classes_), index=1)
chest_pain = st.selectbox("Chest pain type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
                          index=3)
exang = st.selectbox("Exercise-induced angina", list(encoders["ExerciseAngina"].classes_))
thal = st.selectbox("Thal (thallium stress test result)", ["Normal", "Fixed Defect", "Reversible Defect"])

if st.button("Predict"):
    try:
        user_input = pd.DataFrame({"Sex": [sex], "ChestPain": [chest_pain],
                                   "ExerciseAngina": [exang], "Thal": [thal]})[INPUT_COLUMNS]

        # Validate: every value must be a category the encoder was trained on
        for col in INPUT_COLUMNS:
            unknown = set(user_input[col]) - set(encoders[col].classes_)
            if unknown:
                raise ValueError(f"Invalid value for {col}: {sorted(unknown)}")

        encoded = user_input.copy()
        for col in INPUT_COLUMNS:
            encoded[col] = encoders[col].transform(user_input[col])

        pred = int(model.predict(encoded)[0])
        if pred == 1:
            st.error("Prediction: **HEART DISEASE LIKELY**")
        else:
            st.success("Prediction: **NO HEART DISEASE LIKELY**")

        with st.expander("Show the encoded feature vector sent to the model"):
            st.dataframe(encoded)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Educational demo trained on the UCI Cleveland heart disease data (about 300 patients). "
           "Not a medical diagnostic tool.")
