import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

print("🔁 Script started...")

# ✅ Load data
try:
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    print("✅ Data loaded successfully.")
except Exception as e:
    print("❌ Error loading data:", e)
    exit()

# ✅ Label column is "class" in your data
df['class'] = df['class'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# ✅ One-hot encode categorical features
df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])

# ✅ Encode class as integers
df['class'] = LabelEncoder().fit_transform(df['class'])

# ✅ Separate features and labels
try:
    X = df.drop(columns=['class', 'difficulty_level'])  # if difficulty_level exists
except:
    X = df.drop(columns=['class'])
y = df['class']

# ✅ Simulate 30% unlabeled
np.random.seed(42)
mask = np.random.rand(len(y)) < 0.3
y_unlabeled = y.copy()
y_unlabeled[mask] = -1

X_labeled = X[y_unlabeled != -1]
y_labeled = y[y_unlabeled != -1]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

# ✅ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train model
clf = SVC()
clf.fit(X_train_scaled, y_train)
print("✅ SVM trained.")

# ✅ Predict and evaluate
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"🎯 Accuracy: {acc:.4f}")

# ✅ Save model and scaler
joblib.dump(clf, "clf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ✅ Plot and save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Classical Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_classical.png")
plt.show()
print("✅ Model and matrix saved.")
