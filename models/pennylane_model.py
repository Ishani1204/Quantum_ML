import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane.optimize import AdamOptimizer

print("⚡ Fast PennyLane Training Started...\n")

# 📁 Load data
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
df = pd.concat([df_train, df_test], ignore_index=True)

# 🏷️ Relabel class column
df['class'] = df['class'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
df['class'] = LabelEncoder().fit_transform(df['class'])

# 🧪 Simulate 30% unlabeled
np.random.seed(42)
mask = np.random.rand(len(df)) < 0.3
df_unlabeled = df.copy()
df_unlabeled.loc[mask, 'class'] = -1

# 🎯 Prepare labeled data
df_labeled = df_unlabeled[df_unlabeled['class'] != -1].reset_index(drop=True)
X = df_labeled.drop(columns=['class'])
y = df_labeled['class']

# 🔤 Encode categoricals
X = pd.get_dummies(X)
X = X.loc[:, (X != 0).any(axis=0)]  # Remove all-zero cols if any

# ⚖️ Scale and reduce with PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

# ⚠️ Use smaller dataset for speed
X_small, _, y_small, _ = train_test_split(X_pca, y, train_size=200, stratify=y, random_state=42)

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42, stratify=y_small
)

# ⚛️ Quantum circuit
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(params, x):
    qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def circuit(params, x):
    return quantum_circuit(params, x)

def predict_batch(params, X):
    return np.array([(circuit(params, x) + 1) / 2 for x in X])

def binary_crossentropy_loss(y_true, y_pred):
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy(y_true, y_pred):
    return np.mean(y_true == (y_pred > 0.5).astype(int))

# 🧠 Training setup
params = np.random.uniform(-np.pi, np.pi, size=(4, n_qubits, 3))
opt = AdamOptimizer(stepsize=0.02)
epochs = 25
acc_history = []

y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

for epoch in range(epochs):
    params = opt.step(lambda v: binary_crossentropy_loss(y_train_np, predict_batch(v, X_train)), params)
    acc = accuracy(y_train_np, predict_batch(params, X_train))
    acc_history.append(acc)
    print(f"Epoch {epoch+1:02d}: Train Accuracy = {acc:.4f}")

# ✅ Final test accuracy
y_pred = predict_batch(params, X_test)
final_acc = accuracy_score(y_test_np, (y_pred > 0.5).astype(int))
print(f"\n✅ Final Test Accuracy: {final_acc:.4f}")
print("🔮 Sample Predictions:", y_pred[:5])

# 📈 Plot training accuracy
plt.plot(acc_history)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.tight_layout()
plt.show()
