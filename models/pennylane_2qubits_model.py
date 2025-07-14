import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

print("\n🚀 PennyLane Quantum Model (2 Qubits) is starting...\n")

# Create results directory
os.makedirs("results", exist_ok=True)

# Load labeled training data
train_data = pd.read_csv("data/train.csv")

# Separate features and labels
X = train_data.drop(columns=["class"])
y = train_data["class"]

# Encode categorical features
X_encoded = X.copy()
for col in X_encoded.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Use 2 qubits → 2 PCA features
n_qubits = 2
pca = PCA(n_components=n_qubits)
X_reduced = pca.fit_transform(X_scaled)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_reduced, y_encoded, test_size=0.2, random_state=42)

# Quantum device (2 qubits)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def qnn_model(weights, inputs):
    return circuit(inputs, weights)

def cost(weights, features, labels):
    predictions = [qnn_model(weights, x) for x in features]
    predictions = np.stack(predictions)
    return np.mean((predictions - labels) ** 2)

# Initialize weights
np.random.seed(42)
weights = np.random.randn(2, n_qubits, requires_grad=True)

# Train the model
opt = qml.GradientDescentOptimizer(stepsize=0.3)
epochs = 25
train_losses = []
val_losses = []

for epoch in range(epochs):
    weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
    train_loss = cost(weights, X_train, y_train)
    val_loss = cost(weights, X_val, y_val)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# Load unlabeled test data
test_data = pd.read_csv("data/test.csv")
test_encoded = test_data.copy()

# Encode test categorical features
for col in test_encoded.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    test_encoded[col] = le.fit_transform(test_encoded[col])

# Standardize + PCA
X_test_scaled = scaler.transform(test_encoded)
X_test_reduced = pca.transform(X_test_scaled)

# Predict using trained QNN
y_pred_continuous = [qnn_model(weights, x) for x in X_test_reduced]
y_pred = [int(pred >= 0.5) for pred in y_pred_continuous]
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Save predictions
results_df = test_data.copy()
results_df["Predicted_Label"] = y_pred_labels
results_df.to_csv("results/predictions.csv", index=False)
print("\n✅ Predictions saved in 'results/predictions.csv'")

# Show loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", marker='s')
plt.title("QNN Loss Curve (2 Qubits)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()