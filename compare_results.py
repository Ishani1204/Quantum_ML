import matplotlib.pyplot as plt

# 🔢 Manually enter accuracy values here (between 0 and 1)
accuracies = {
    "Classical": 0.87,
    "Qiskit": 0.55,
    "PennyLane": 0.52,
}

# 🖨️ Print accuracies
print("\n📊 Accuracy Comparison:")
for model, acc in accuracies.items():
    print(f"{model:10}: {acc:.2f}")

# 📈 Plotting the bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['green', 'purple', 'orange'])
plt.title('🔬 Accuracy Comparison of ML Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# 🏷️ Adding accuracy labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height:.2f}", ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
