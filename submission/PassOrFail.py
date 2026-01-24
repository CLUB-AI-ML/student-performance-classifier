# ========================
# IMPORT LIBRARIES
# ========================
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========================
# LOAD DATASET
# ========================
df = pd.read_csv("data/dataset.csv")

X = df.drop("result", axis=1)
y = df["result"]

# ========================
# SPLIT DATA
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# SCALE DATA
# ========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# TRAIN MODEL
# ========================
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ========================
# EVALUATE MODEL
# ========================
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ========================
# SAVE MODEL
# ========================
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")

print("\nModel & scaler saved successfully")

# ========================
# USER INPUT PREDICTION
# ========================
print("\nEnter Student Details for Result Prediction")

study_hours = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage: "))
sleep_hours = float(input("Sleep hours per day: "))
past_grade = float(input("Past grade (out of 100): "))
practice_tests = int(input("Practice tests taken: "))

new_student = np.array([[study_hours, attendance, sleep_hours, past_grade, practice_tests]])
new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("Student is likely to PASS")
else:
    print("Student is likely to FAIL")
