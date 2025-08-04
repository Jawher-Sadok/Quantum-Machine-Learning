import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Load Iris dataset
iris = load_iris()
X = iris.data[:100, [0, 2]]  # Select Setosa and Versicolor, use sepal length and petal length
y = iris.target[:100]  # Binary labels (0: Setosa, 1: Versicolor)

# Split and normalize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define quantum feature map (2 qubits for 2 features)
feature_map = ZZFeatureMap(feature_dimension=2, reps=1, entanglement='full')

# Set up quantum kernel with Aer simulator
backend = AerSimulator()
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Train QSVM
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)

# Evaluate model
y_pred = qsvc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Visualize decision boundary
def plot_decision_boundary(X, y, model, scaler):
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
    plt.xlabel('Sepal Length (normalized)')
    plt.ylabel('Petal Length (normalized)')
    plt.title('QSVM Decision Boundary on Iris Dataset')
    plt.show()

# Plot results
plot_decision_boundary(X_train, y_train, qsvc, scaler)
