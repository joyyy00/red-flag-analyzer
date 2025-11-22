# Logistic Regression + TF-IDF pipeline

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


# Dataset
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
    vectorizer = TfidfVectorizer(max_features=5000, stop_words= 'english', ngram_range=(1,2))

    #convert sparse matrix to to dense array (small dataset) 
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # feature scaling
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_scaled = scaler.transform(X_test_tfidf)

    if not isinstance(X_train_scaled, np.ndarray):
        X_train_tfidf = X_train_scaled.toarray()
        X_test_tfidf = X_test_scaled.toarray()

    return X_train_tfidf, X_test_tfidf, vectorizer


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
def crossValidate(X, y, k, lr=0.1, num_iter=1000):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train your custom model
        model = CustomLogisticRegression(lr=lr, num_iter=num_iter)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        accuracy_scores.append(acc)

    return np.mean(accuracy_scores), np.std(accuracy_scores)
    

class CustomLogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        '''
        Hyperparameters:
        lr (learning rate): too large will cause divergence, too small will cause slow covergence
        num_iter: number of iterations (epochs)
        weighets: 
        bias: 
        '''
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        '''
        Logistic function - Sigmoid
        maps real numbers between range of 0 and 1
        '''
        z = np.clip(z, -20, 20)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        '''
        model training 
        '''        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # calcualte class weights
        classes = np.unique(y)
        class_counts = np.array([np.sum(y==c) for c in classes])
        class_weight = {c: n_samples/(len(classes) * count) for c, count in zip(classes, class_counts)}
        weights_vector = np.array([class_weight[label] for label in y])

        # calculate gradients using the whole training set each iteration (change to stochastic for bigger dataset) 
        for _ in range(self.num_iter):
            # multiply vectors X and weight
            # apply sigmoid to get prob
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # weighted gradient loss
            # w.r.t. weights
            # dw = (1/n_samples) * X.T @ ((y_pred - y) * weights_vector)
            dw = (X.T @ ((y_pred - y) * weights_vector)) / np.sum(weights_vector)
            # w.r.t. bias
            # db = (1/n_samples) * np.sum((y_pred - y) * weights_vector)
            db = np.sum((y_pred - y) * weights_vector) / np.sum(weights_vector)

            # gradient descent
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # calculate loss every 100 iterations
            if _ % 100 == 0:
                loss = calculateNegativeLogLikelihood(X, y, self.weights, self.bias)
                print(f"Iter {_}: loss={loss:.4f}")
    

    def predict(self, X, threshold = 0.4):
        z = X @ self.weights + self.bias
        y_pred_probs = self.sigmoid(z)
        return np.where(y_pred_probs >= threshold, 1, 0)


def main():
    # Load data
    X_train, X_test, y_train, y_test = loadData()

    # TF-IDF vectorizer
    X_train_tfidf, X_test_tfidf, vectorizer = tfidf(X_train, X_test)

    # Convert to numpy arrays for easier slicing in k-fold
    X = np.vstack([X_train_tfidf, X_test_tfidf])
    y = np.hstack([y_train.values, y_test.values])

    #cross-validation
    print("="*60)
    mean_acc, std_acc = crossValidate(X, y, k=5)
    print(f"Cross-val Accuracy: {mean_acc: 0.3f} ± {std_acc: 0.3f}")

    # Train via logistic regression 
    model = CustomLogisticRegression(lr=0.01, num_iter=2000)
    model.fit(X_train_tfidf, y_train.values)

    # Predict 
    y_pred = model.predict(X_test_tfidf)   

    # Evaluation
    print("Shape of TF-IDF matrix: ", X_train_tfidf.shape)
    print("\nAccuracy: ", accuracy_score(y_test, y_pred))
    # classification_report: precision, recall, F1 per class
    print("\nClassificaton report: \n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
    
    logits = X_test_tfidf.dot(model.weights) + model.bias
    y_prob_custom = 1 / (1 + np.exp(-logits))
    print("Custom model prob summary:", 
        np.percentile(y_prob_custom, [0, 25, 50, 75, 100]))


    print("="*60)


if __name__ == "__main__":
    main()
    print("Done")

