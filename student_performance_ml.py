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
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==============================
# 1. Load Dataset
# ==============================
print("Loading dataset...")
if not os.path.exists("StudentsPerformance.csv"):
    raise FileNotFoundError("❌ Dataset 'StudentsPerformance.csv' not found in project folder.")

df = pd.read_csv("StudentsPerformance.csv")
print(f"✅ Dataset loaded with shape: {df.shape}\n")

# ==============================
# 2. Preprocessing
# ==============================
df.columns = [c.strip().lower().replace("/", "_").replace(" ", "_") for c in df.columns]

# Verify required columns
required_cols = {"math_score", "reading_score", "writing_score"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"❌ Missing required columns: {required_cols - set(df.columns)}")

df["average_score"] = df[["math_score", "reading_score", "writing_score"]].mean(axis=1)
df["pass_fail"] = np.where(df["average_score"] >= 50, 1, 0)

X = df.drop(columns=["pass_fail", "average_score"])
y = df["pass_fail"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Encoder compatibility for sklearn versions
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
# 3. Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Data split done\n")

# ==============================
# 4. Model Training
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

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}
    print(f"{name} Results: Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | ROC={roc:.3f}\n")

    if acc > best_score:
        best_score = acc
        best_model_name = name
        best_pipe = pipe

# ==============================
# 5. Save Best Model
# ==============================
print(f"🏆 Best model: {best_model_name} with accuracy {best_score:.3f}")

model_path = "best_student_model.pkl"
joblib.dump(best_pipe, model_path)
print(f"✅ Best model saved as '{model_path}'")
