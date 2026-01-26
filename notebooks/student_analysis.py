import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import os

# --- Configuration ---
DATA_PATH = '../dataset/student_data.csv'
SUBMISSION_DIR = '../submission'
EDA_DIR = '../notebooks/eda_plots'

# Ensure directories exist
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# --- Phase 2: Exploration & Analysis ---
print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    # Handle case where script is run from notebooks directory vs root
    if os.path.exists('dataset/student_data.csv'):
        df = pd.read_csv('dataset/student_data.csv')
        DATA_PATH = 'dataset/student_data.csv'
        SUBMISSION_DIR = 'submission'
        EDA_DIR = 'notebooks/eda_plots'
        # Re-create dirs with new paths if needed
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        os.makedirs(EDA_DIR, exist_ok=True)
    else:
        raise

print(f"Data loaded: {df.shape}")

# Visualizations (Saved to EDA_DIR)
# Histograms
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'histograms.png'))
plt.close()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(EDA_DIR, 'correlation_heatmap.png'))
plt.close()

# Study Hours vs Pass/Fail
plt.figure(figsize=(8, 6))
sns.boxplot(x='result', y='study_hours', data=df)
plt.title('Study Hours vs Pass/Fail')
plt.savefig(os.path.join(EDA_DIR, 'study_hours_vs_result.png'))
plt.close()

# --- Phase 3: Building the Model ---
print("Building model...")
X = df.drop('result', axis=1)
y = df['result']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# --- Phase 4: Internal Evaluation ---
print("Evaluating model...")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Check thresholds
if accuracy >= 0.85:
    print("PASS: Accuracy requirement met.")
else:
    print("FAIL: Accuracy requirement NOT met.")

if f1 >= 0.80:
    print("PASS: F1 Score requirement met.")
else:
    print("FAIL: F1 Score requirement NOT met.")

# --- Phase 5: Validation & Submission ---
print("Generating submission artifacts...")

# 1. Model Performance Report
report = classification_report(y_test, y_pred)
with open(os.path.join(SUBMISSION_DIR, 'model_performance.txt'), 'w') as f:
    f.write("Model Performance Metrics\n")
    f.write("=========================\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# 2. Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(SUBMISSION_DIR, 'confusion_matrix.png'))
plt.close()

# 3. ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(SUBMISSION_DIR, 'roc_curve.png'))
plt.close()

print("Submission files generated successfully.")
