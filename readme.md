Quantum Machine Learning for Intrusion Detection
------------------------------------------------
Overview
This project explores the use of Quantum Machine Learning (QML) for classifying network intrusion data , using both classical and quantum models (Qiskit and PennyLane).
It predicts labels on unlabeled NSL-KDD test data. The models are trained and evaluated on labeled training data.

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
