import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
# ==============================
# 1. Load Trained Model
# ==============================
model_path = os.path.join(os.path.dirname(__file__), "best_student_model.pkl")
best_pipe = joblib.load(model_path)



columns = ['gender','race/ethnicity','parental_level_of_education','lunch',
           'test_preparation_course','math_score','reading_score','writing_score']

# ==============================
# 2. Streamlit UI
# ==============================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor & AI Insights")
st.markdown("Enter student details in the sidebar and get predictions with improvement suggestions.")

st.sidebar.header("Enter Student Details")

student_data = {
    'gender': st.sidebar.selectbox("Gender", ['female','male']),
    'race/ethnicity': st.sidebar.selectbox("Race/Ethnicity", ['group A','group B','group C','group D','group E']),
    'parental_level_of_education': st.sidebar.selectbox(
        "Parental Education", 
        ["bachelor's degree","some college","master's degree","associate's degree","high school","some high school"]
    ),
    'lunch': st.sidebar.selectbox("Lunch", ['standard','free/reduced']),
    'test_preparation_course': st.sidebar.selectbox("Test Preparation", ['none','completed']),
    'math_score': st.sidebar.slider("Math Score", 0, 100, 50),
    'reading_score': st.sidebar.slider("Reading Score", 0, 100, 50),
    'writing_score': st.sidebar.slider("Writing Score", 0, 100, 50)
}

# ==============================
# 3. Prediction Button
# ==============================
if st.button("Predict Performance"):
    # Ensure all expected columns are present
    safe_data = {col: student_data.get(col, 0) for col in columns}
    df_input = pd.DataFrame([safe_data])

    # Predict probability and label
    pred_proba = best_pipe.predict_proba(df_input)[0][1]
    pred_label = "pass" if pred_proba >= 0.5 else "fail"

    # Generate AI-based suggestions
    if pred_label == "fail":
        suggestions = [
            "Focus on weaker subjects (especially Math or Reading).",
            "Consider a structured study plan and consistent schedule.",
            "Engage with peer study groups or mentorship programs.",
            "Seek feedback from teachers regularly."
        ]
    else:
        suggestions = [
            "Maintain consistent study habits.",
            "Continue practice tests to retain strength.",
            "Explore advanced or extracurricular learning challenges."
        ]

    # ==============================
    # 4. Display Results
    # ==============================
    st.subheader("Prediction Results")
    st.write(f"**Predicted Probability of Passing:** {pred_proba:.2f}")
    st.write(f"**Predicted Label:** {pred_label}")

    st.subheader("Improvement Suggestions")
    for s in suggestions:
        st.write(f"- {s}")

# ==============================
# 5. Optional Notes / Footer
# ==============================
st.markdown("---")
st.markdown("Developed by Adithya | ML model trained on Kaggle Student Performance dataset")
