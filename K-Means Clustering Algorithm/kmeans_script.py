
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Set random seed for reproducibility
np.random.seed(42)

class KMeansScratch:
    def __init__(self, k=3, max_iters=100, tol=0.0001):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.inertia_ = 0

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b)**2))

    def fit(self, X):
        # 1. Initialize Centroids randomly from the data points
        indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iters):
            # 2. Assign clusters
            clusters = [[] for _ in range(self.k)]
            labels = []
            
            for point in X:
                distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
                closest_centroid_index = np.argmin(distances)
                clusters[closest_centroid_index].append(point)
                labels.append(closest_centroid_index)
            
            # 3. Update centroids
            prev_centroids = self.centroids.copy()
            for i in range(self.k):
                if clusters[i]: # Avoid empty cluster division
                    self.centroids[i] = np.mean(clusters[i], axis=0)
            
            # Check for convergence
            shift = np.sum([self.euclidean_distance(prev_centroids[i], self.centroids[i]) for i in range(self.k)])
            if shift < self.tol:
                break
                
        # Calculate Inertia (Sum of squared distances)
        self.inertia_ = 0
        for i, point in enumerate(X):
            self.inertia_ += self.euclidean_distance(point, self.centroids[labels[i]])**2
            
        return np.array(labels)

def main():
    print("Generating synthetic dataset...")
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Visualize Initial State (Random Centroids logic for demo usually implies before training, 
    # but here we'll just show the raw data first)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], s=50, c='gray', alpha=0.5)
    plt.title("Initial Dataset")
    plt.savefig('initial_state.png')
    print("Saved initial_state.png")

    # --- Run K-Means for a specific K (e.g., K=4) to visualize final result ---
    print("Running K-Means (K=4)...")
    kmeans = KMeansScratch(k=4)
    labels = kmeans.fit(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    plt.title("K-Means Clustering Results (K=4)")
    plt.legend()
    plt.savefig('final_clusters.png')
    print("Saved final_clusters.png")

    # --- Elbow Method ---
    print("Calculating Inertia for different K values (Elbow Method)...")
    inertias = []
    K_range = range(1, 11)
    
    for k in K_range:
        km = KMeansScratch(k=k)
        km.fit(X)
        inertias.append(km.inertia_)
        
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.xticks(K_range)
    plt.grid(True)
    plt.savefig('elbow_plot.png')
    print("Saved elbow_plot.png")
    
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
