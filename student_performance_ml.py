"""
Project: Predict Student Performance using ML & Generate AI-based Insights
Author: Adithya
"""

# ==============================
# 1. Import Dependencies
# ==============================
import pandas as pd
import numpy as np
import os
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# import shap  # Optional: uncomment to enable SHAP visualizations

# ==============================
# 2. Load Dataset
# ==============================
print("Loading dataset...")
df = pd.read_csv("StudentsPerformance.csv")
print(f"✅ Dataset loaded with shape: {df.shape}\n")

# ==============================
# 3. Preprocessing
# ==============================
# Clean column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Target variable: pass/fail
df["average_score"] = df[["math_score", "reading_score", "writing_score"]].mean(axis=1)
df["pass_fail"] = np.where(df["average_score"] >= 50, 1, 0)

# Features and target
X = df.drop(columns=["pass_fail", "average_score"])
y = df["pass_fail"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ✅ Compatible OneHotEncoder for modern scikit-learn
if sklearn.__version__ >= "1.4":
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', ohe, categorical_features)
    ]
)

# ==============================
# 4. Split Data
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Data split done\n")

# ==============================
# 5. Model Training
# ==============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}
best_model_name, best_score, best_pipe = None, 0, None

for name, clf in models.items():
    print(f"🔹 Training {name}...")
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    }

    print(f"{name} Results:")
    print(f"  Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {roc:.3f}\n")

    if acc > best_score:
        best_score = acc
        best_model_name = name
        best_pipe = pipe

if best_pipe is not None:
    print("✅ Best Model:", best_model_name, "with Accuracy:", round(best_score, 3))
else:
    print("❌ No model trained successfully.")

# ==============================
# 6. Save Model
# ==============================
if best_pipe is not None:
    save_path = os.path.join(os.path.dirname(__file__), "best_student_model.pkl")
    joblib.dump(best_pipe, save_path)
    print(f"💾 Model saved successfully to: {os.path.abspath(save_path)}\n")

# ==============================
# 7. Column-Safe AI-Based Insights
# ==============================
def generate_insights(student_data: dict):
    if best_pipe is None:
        return {"error": "Model not available."}

    # Ensure all expected columns are present
    expected_cols = X.columns.tolist()
    safe_data = {}
    for col in expected_cols:
        safe_data[col] = student_data.get(col, 0)  # default 0 if missing

    df_input = pd.DataFrame([safe_data])
    pred_proba = best_pipe.predict_proba(df_input)[0][1]
    pred_label = "pass" if pred_proba >= 0.5 else "fail"

    suggestions = []
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

    return {
        "predicted_probability": round(pred_proba, 2),
        "predicted_label": pred_label,
        "suggestions": suggestions
    }

# ==============================
# 8. Example Run
# ==============================
sample_student = {
    'gender': 'female',
    'race/ethnicity': 'group B',
    'parental_level_of_education': "bachelor's degree",
    'lunch': 'standard',
    'test_preparation_course': 'none',
    'math_score': 52,
    'reading_score': 48,
    'writing_score': 51
}

if best_pipe is not None:
    print("🧠 Generating insights for a sample student...")
    insights = generate_insights(sample_student)
    print("Example insights:\n", insights)
else:
    print("❌ Model training failed — cannot generate insights.")

# ==============================
# 9. Optional SHAP Analysis
# ==============================
"""
# Uncomment this section if you want SHAP visualizations (requires GUI or Jupyter)
print("Computing SHAP values (may take some time)...")
X_test_trans = preprocessor.transform(X_test)
explainer = shap.Explainer(best_pipe.named_steps['classifier'])
shap_values = explainer(X_test_trans)
shap.summary_plot(shap_values, features=X_test_trans, show=True)
"""
