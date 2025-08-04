import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Generate synthetic spiral dataset
def generate_spiral_dataset(n_samples=100, noise=0.1):
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, n_samples // 2)
    
    # Class 0: First spiral
    r1 = t / (4 * np.pi)  # Radius increases linearly
    x1 = r1 * np.cos(t) + np.random.normal(0, noise, n_samples // 2)
    y1 = r1 * np.sin(t) + np.random.normal(0, noise, n_samples // 2)
    
    # Class 1: Second spiral, offset by pi
    r2 = t / (4 * np.pi)
    x2 = r2 * np.cos(t + np.pi) + np.random.normal(0, noise, n_samples // 2)
    y2 = r2 * np.sin(t + np.pi) + np.random.normal(0, noise, n_samples // 2)
    
    # Combine data
    X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    return X, y

# Generate dataset
X, y = generate_spiral_dataset(n_samples=300, noise=0.0)

# Split and normalize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classical SVM with RBF kernel
classical_svm = SVC(kernel='rbf', random_state=42)
classical_svm.fit(X_train, y_train)
y_pred_classical = classical_svm.predict(X_test)
classical_accuracy = accuracy_score(y_test, y_pred_classical)
print(f"Classical SVM Accuracy: {classical_accuracy:.2f}")

# Define quantum feature map (2 qubits for 2 features)
feature_map = ZZFeatureMap(feature_dimension=2, reps=7, entanglement='full')

# Set up quantum kernel with Aer simulator
backend = AerSimulator()
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Train QSVM
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)

# Evaluate QSVM
y_pred_qsvm = qsvc.predict(X_test)
qsvm_accuracy = accuracy_score(y_test, y_pred_qsvm)
print(f"QSVM Accuracy: {qsvm_accuracy:.2f}")

# Visualize decision boundaries
def plot_decision_boundary(X, y, model, scaler, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Transform mesh points
    mesh_points = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1 (normalized)')
    plt.ylabel('Feature 2 (normalized)')
    plt.title(title)
    plt.show()

# Plot results for both models
plot_decision_boundary(X_train, y_train, classical_svm, scaler, 'Classical SVM Decision Boundary')
plot_decision_boundary(X_train, y_train, qsvc, scaler, 'QSVM Decision Boundary')