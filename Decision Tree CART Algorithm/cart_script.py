
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

# --- Part 1: CART Implementation from Scratch ---

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class CARTFromScratch:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        # Handle if X is dataframe
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        feat_idxs = np.random.choice(n_features, n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        # Grow children
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
             return Node(value=self._most_common_label(y))
             
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, split_thresh):
        # Parent Gini
        parent_gini = self._gini(y)
        
        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
            
        # Weighted avg child Gini
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l/n) * e_l + (n_r/n) * e_r
        
        # Gain = Parent_Gini - Children_Gini (Maximizing gain reduces impurity)
        return parent_gini - child_gini
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _gini(self, y):
        # Gini Impurity = 1 - sum(p_i^2)
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - sum(probabilities**2)
        return gini
    
    def _most_common_label(self, y):
        return np.bincount(y).argmax()
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# --- Part 2: Main Execution & Comparison ---

def evaluate_model(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted') # Sensitivity (Recall)
    
    print(f"--- {name} Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    return acc, prec, rec

def main():
    print("Loading Breast Cancer Dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []

    # 1. Custom CART (From Scratch)
    print("\nTraining Custom CART (Scratch)...")
    start = time.time()
    cart_scratch = CARTFromScratch(max_depth=5)
    cart_scratch.fit(X_train, y_train)
    y_pred_cart = cart_scratch.predict(X_test)
    end = time.time()
    print(f"Training Time: {end-start:.4f}s")
    metrics = evaluate_model(y_test, y_pred_cart, "Custom CART")
    results.append(["Custom CART", *metrics])
    
    # 2. Scikit-Learn 'Entropy' (Proxy for ID3/C4.5)
    # Note: Sklearn uses an optimized CART algorithm but with 'entropy' criterion 
    # it behaves similarly to C4.5 in terms of information gain logic.
    print("\nTraining Sklearn Entropy (ID3/C4.5 Proxy)...")
    clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    clf_entropy.fit(X_train, y_train)
    y_pred_entropy = clf_entropy.predict(X_test)
    metrics = evaluate_model(y_test, y_pred_entropy, "ID3/C4.5 (Sklearn)")
    results.append(["ID3/C4.5 (Sklearn)", *metrics])
    
    # 3. Scikit-Learn 'Gini' (Standard CART)
    print("\nTraining Sklearn CART (Gini)...")
    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    clf_gini.fit(X_train, y_train)
    y_pred_gini = clf_gini.predict(X_test)
    metrics = evaluate_model(y_test, y_pred_gini, "CART (Sklearn)")
    results.append(["CART (Sklearn)", *metrics])
    
    # Save results to file
    with open("metrics_output.txt", "w") as f:
        f.write("Algorithm,Accuracy,Precision,Recall\n")
        for res in results:
            f.write(f"{res[0]},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f}\n")
            
    print("\nAnalysis Complete. Results saved to metrics_output.txt")

if __name__ == "__main__":
    main()
