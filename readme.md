Quantum Machine Learning for Intrusion Detection
------------------------------------------------
Overview
This project explores the use of Quantum Machine Learning (QML) for classifying network intrusion data. Using both classical and quantum models (Qiskit and PennyLane).
It predicts labels on unlabeled NSL-KDD test data. The models are trained and evaluated on labeled training data.

Project Structure
Below is the structure of the project directory:

Quantum_ML/
├── data/
│   ├── train.csv              # Labeled dataset
│   └── test.csv               # Unlabeled dataset
│
├── models/
│   ├── classical_model.py     # Classical ML implementation
│   ├── qiskit_model.py        # Qiskit-based Quantum Model
│   ├── pennylane_model.py     # PennyLane Quantum Model (multi-qubit)
│   └── pennylane_2qubit_model.py # Simplified 2-qubit model
│
├── results/
│   ├── predictions.csv        # Final predictions
│   ├── confusion_matrix.png   # Confusion matrix plot
│   ├── loss_curve.png         # Loss curve
│   └── comparison_chart.png   # Accuracy comparison
│
├── saved_models/
│   ├── clf_model.pkl
│   ├── scaler.pkl
│   ├── pennylane_labels.npy
│   └── pennylane_predictions.npy
│
├── requirements.txt
├── README.md
└── .gitignore

Dataset
The dataset used is a processed version of the NSL-KDD intrusion detection dataset.
- train.csv: Used for model training and validation.
- test.csv: Used for prediction on unseen/unlabeled data.
  
Models Used
Model	Library	Approx.       Accuracy
Classical ML scikit-learn	  ~80%
Qiskit QNN Qiskit	          ~50-55%
PennyLane QNN PennyLane	    ~55–60%

Features
- Preprocessing using Label Encoding and StandardScaler
- Dimensionality reduction with PCA
- Classical SVM or RandomForest baseline
- Qiskit and PennyLane-based quantum neural networks
- Loss and accuracy visualization
  
How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run individual models:
   python models/classical_model.py
   python models/qiskit_model.py
   python models/pennylane_model.py

Requirements
- Python 3.8+
- Qiskit
- PennyLane
- pandas, scikit-learn, matplotlib

Future Work
- Use more qubits and deeper circuits
- Hybrid classical + quantum models
- Deploy on real quantum devices
