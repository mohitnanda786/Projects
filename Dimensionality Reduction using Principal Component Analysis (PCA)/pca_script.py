
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Set style
sns.set(style='whitegrid')

def main():
    print("Starting PCA Analysis...")

    # 1. Load Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dataset loaded. Shape: {X.shape}")

    # 2. Standardization
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    print("Data standardized.")

    # 3. Determine number of components
    pca_full = PCA()
    pca_full.fit(X_std)
    
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\nExplained Variance Ratio per component:", explained_variance)
    print("Cumulative Variance:", cumulative_variance)

    # Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 5), cumulative_variance, marker='o', linestyle='--')
    plt.title('Scree Plot / Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig('scree_plot.png')
    print("Saved scree_plot.png")

    # 4. Apply PCA with 2 components (usually sufficient for visualization and covers ~95% variance)
    n_components = 2
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    print(f"\nApplied PCA with {n_components} components.")

    # 5. Visualize
    plt.figure(figsize=(10, 6))
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('pca_visualization.png')
    print("Saved pca_visualization.png")

    # 6. Evaluation (Machine Learning Task)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    # Train Logicstic Regression on original features
    start_time = time.time()
    clf_original = LogisticRegression()
    clf_original.fit(X_train, y_train)
    original_time = time.time() - start_time
    original_acc = accuracy_score(y_test, clf_original.predict(X_test))

    # Train Logistic Regression on PCA features
    start_time = time.time()
    clf_pca = LogisticRegression()
    clf_pca.fit(X_train_pca, y_train_pca)
    pca_time = time.time() - start_time
    pca_acc = accuracy_score(y_test_pca, clf_pca.predict(X_test_pca))

    print("\n--- Performance Evaluation ---")
    print(f"Original Data (4 features) - Accuracy: {original_acc:.4f}, Training Time: {original_time:.6f} sec")
    print(f"PCA Reduced (2 features)   - Accuracy: {pca_acc:.4f}, Training Time: {pca_time:.6f} sec")

    # Save metrics to text file for easy reading
    with open("pca_output.txt", "w") as f:
        f.write("Explained Variance Ratio:\n" + str(explained_variance) + "\n\n")
        f.write("Cumulative Variance:\n" + str(cumulative_variance) + "\n\n")
        f.write(f"Original Data Accuracy: {original_acc:.4f}\n")
        f.write(f"PCA Data Accuracy: {pca_acc:.4f}\n")

if __name__ == "__main__":
    main()
