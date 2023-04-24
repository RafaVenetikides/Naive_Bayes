from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_samples, n_features = X_train.shape
n_classes = len(np.unique(y_train))
priors = np.zeros(n_classes)
for c in range(n_classes):
    priors[c] = np.sum(y_train == c) / float(n_samples)

eps = 1e-8 # constante para evitar divisão por zero
cond_probs = np.zeros((n_classes, n_features, 17))
for c in range(n_classes):
    for f in range(n_features):
        values, counts = np.unique(X_train[y_train == c, f], return_counts=True)
        for i, v in enumerate(values):
            cond_probs[c, f, int(v)] = (counts[i] + eps) / (np.sum(counts) + eps*10)

y_pred = np.zeros(len(y_test))
for i, x in enumerate(X_test):
    posteriors = np.zeros(n_classes)
    for c in range(n_classes):
        likelihood = 1.0
        for f in range(n_features):
            likelihood *= cond_probs[c, f, round(x[f])]
        posteriors[c] = priors[c] * likelihood
    y_pred[i] = np.argmax(posteriors)

accuracy = np.sum(y_pred == y_test) / float(len(y_test))
print("Accuracy: {:.2f}%".format(accuracy*100))
