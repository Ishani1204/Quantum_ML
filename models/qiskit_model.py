import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# Load data
df = pd.read_csv("data/train.csv")
df['class'] = df['class'].apply(lambda x: 1 if x != 'normal' else 0)

# Encode categorical
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("class", axis=1)
y = df["class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dataset for quantum processing
X_small, _, y_small, _ = train_test_split(X_scaled, y, train_size=0.2, stratify=y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.25, stratify=y_small, random_state=42)

# Qiskit VQC Setup
feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1)
ansatz = RealAmplitudes(num_qubits=X_train.shape[1], reps=1)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=None,
    quantum_instance=Sampler(),
)

print("\n⚛️ Training Qiskit VQC Model...")
vqc.fit(X_train, y_train)
y_pred = vqc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Qiskit VQC Accuracy: {accuracy * 100:.2f}%")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix - Qiskit VQC")
plt.show()
