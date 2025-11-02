import os 
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# 1. Load Trained Model
# ==============================
model_path = os.path.join(os.path.dirname(__file__), "best_student_model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please run 'student_performance_ml.py' first to train and save the model.")
    st.stop()

best_pipe = joblib.load(model_path)

columns = [
    'gender',
    'race_ethnicity',
    'parental_level_of_education',
    'lunch',
    'test_preparation_course',
    'math_score',
    'reading_score',
    'writing_score'
]

# ==============================
# 2. Streamlit UI
# ==============================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor & AI Insights")
st.markdown("Enter student details in the sidebar and get predictions with improvement suggestions.")

st.sidebar.header("Enter Student Details")

student_data = {
    'gender': st.sidebar.selectbox("Gender", ['female', 'male']),
    'race_ethnicity': st.sidebar.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E']),
    'parental_level_of_education': st.sidebar.selectbox(
        "Parental Education",
        ["bachelor's degree", "some college", "master's degree",
         "associate's degree", "high school", "some high school"]
    ),
    'lunch': st.sidebar.selectbox("Lunch", ['standard', 'free/reduced']),
    'test_preparation_course': st.sidebar.selectbox("Test Preparation", ['none', 'completed']),
    'math_score': int(st.sidebar.slider("Math Score", 0, 100, 50)),
    'reading_score': int(st.sidebar.slider("Reading Score", 0, 100, 50)),
    'writing_score': int(st.sidebar.slider("Writing Score", 0, 100, 50))
}

# ==============================
# 3. Prediction Button
# ==============================
if st.button("Predict Performance"):
    safe_data = {col: student_data.get(col, 0) for col in columns}
    df_input = pd.DataFrame([safe_data])

    numeric_cols = ['math_score', 'reading_score', 'writing_score']
    df_input[numeric_cols] = df_input[numeric_cols].apply(pd.to_numeric, errors='coerce')

    pred_proba = best_pipe.predict_proba(df_input)[0][1]
    pred_label = "pass" if pred_proba >= 0.5 else "fail"

    # Generate AI-based suggestions
    if pred_label == "fail":
        suggestions = [
            "Focus on weaker subjects (especially Math or Reading).",
            "Create a consistent study schedule.",
            "Join peer study or mentoring groups.",
            "Ask for teacher feedback regularly."
        ]
    else:
        suggestions = [
            "Keep up consistent study habits.",
            "Take regular practice tests.",
            "Try challenging topics or advanced courses."
        ]

    # ==============================
    # 4. Display Results
    # ==============================
    st.subheader("Prediction Results")
    st.write(f"**Predicted Probability of Passing:** {pred_proba:.2f}")
    st.write(f"**Predicted Label:** {pred_label}")

    st.subheader("Improvement Suggestions")
    for s in suggestions:
        st.markdown(f"- {s}")

# ==============================
# 5. Footer
# ==============================
st.markdown("---")
st.markdown("ML model trained on Kaggle Student Performance dataset")
