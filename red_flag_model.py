# Logistic Regression + TF-IDF pipeline

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def loadData():
    df = pd.read_csv("reddit_messages_cleaned.csv")
    # use cleaned column for modeling
    texts = df["CleanText"].astype(str)
    labels = df["Label"].map({"green_flag": 0, "red_flag": 1})
    
    # split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size= 0.2, random_state= 42, stratify= labels)

    return X_train, X_test, y_train, y_test


# Converts text to TF-IDF vectors
def tfidf(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words= 'english', ngram_range=(1,1))
    
    #convert sparse matrix to to dense array (small dataset) 
    X_train_tdidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()

    return X_train_tdidf, X_test_tfidf, vectorizer


# Tracks training loss 
def calculateNegativeLogLikelihood(X, y, weights, bias):
  z = X @ weights + bias
  y_hat = np.reciprocal(1+np.exp(-z))

  eps = 1e-15
  y_hat = np.clip(y_hat, eps, 1-eps)

  # calculate negative log likelihood
  nll = -np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
  return nll


# Compute accuracy for each split/fold
def crossValidate(X, y, k, lr=0.1, n_iter=1000):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train your custom model
        model = CustomLogisticRegression(lr=lr, n_iter=n_iter)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        accuracy_scores.append(acc)

    return np.mean(accuracy_scores), np.std(accuracy_scores)
    

class CustomLogisticRegression:
    '''
    Hyperparameters:
    lr (learning rate): too large will cause divergence, too small will cause slow covergence
    n_iter: number of iterations (epochs)
    '''
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        '''
        maps real numbers between range of 0 and 1
        '''
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # computes gradients using the whole training set each iteration (change to stochastic for bigger dataset) 
        for _ in range(self.n_iter):
            # multiply vectors X and weight
            # apply sigmoid to get prob
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # gradient loss
            # w.r.t. weights
            dw = (1/n_samples) * X.T @ (y_pred - y)
            # w.r.t. bias
            db = (1/n_samples) * np.sum(y_pred - y)

            # gradient descent
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        z = X @ self.weights + self.bias
        y_pred_probs = self.sigmoid(z)
        return np.where(y_pred_probs >= 0.5, 1, 0)


def main():
    # Load data
    X_train, X_test, y_train, y_test = loadData()

    # TF-IDF vectorizer
    X_train_tfidf, X_test_tfidf, vectorizer = tfidf(X_train, X_test)

    # Convert to numpy arrays for easier slicing in k-fold
    X = np.vstack([X_train_tfidf, X_test_tfidf])
    y = np.hstack([y_train.values, y_test.values])

    #cross-validation
    mean_acc, std_acc = crossValidate(X, y, k=5)
    print(f"Cross-val Accuracy: {mean_acc: 0.3f} ± {std_acc: 0.3f}")

    # Train via logistic regression 
    model = CustomLogisticRegression(lr=0.1, n_iter=1000)
    model.fit(X_train_tfidf, y_train.values)

    # Predict 
    y_pred = model.predict(X_test_tfidf)

    # Evaluation
    print("="*60)
    print("Shape of TF-IDF matrix: ", X_train_tfidf.shape)
    print("\nAccuracy: ", accuracy_score(y_test, y_pred))
    # classification_report: precision, recall, F1 per class
    print("\nClassificaton report: \n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("="*60)


if __name__ == "__main__":
    main()
    print("Done")

