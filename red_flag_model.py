# Logistic Regression Model + TF-IDF pipeline

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =====================
# Configuration
# =====================
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_K_FEATURES = 15


# =====================
# Data Loading
# =====================
def loadData():
    df = pd.read_csv("reddit_messages_cleaned.csv")

    # use cleaned column for modeling
    texts = df["CleanText"].astype(str)
    labels = df["Label"].map({"green_flag": 0, "red_flag": 1})
    
    # return train and test data
    return train_test_split(
        texts, 
        labels, 
        test_size = TEST_SIZE, 
        random_state = RANDOM_STATE, 
        stratify = labels
        )


# ========================================
# TF-IDF
# ========================================
def buildVectorizer():
    # custom stop words
    stop_words = list(text.ENGLISH_STOP_WORDS)
    custom_list = ['told', 'anymore', 'post', 'stories', 'line', 'messaged', 'happened', 'said', 'line', 'little', 
                   'going', 'tldr', 'edit', 'update', 'reddit', 'truck', 'lmao']
    stop_words.extend(custom_list)

    return TfidfVectorizer(
        max_features = 2000, 
        stop_words = stop_words, 
        ngram_range =(1,2), 
        min_df = 3, 
        token_pattern = r'(?u)\b\w\w+\b|[!?]'
        )


def featureScaling(X_train, X_test):
    vectorizer = buildVectorizer()
    scaler = StandardScaler(with_mean=False)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_scaled = scaler.transform(X_test_tfidf)

    return (
        X_train_scaled.toarray(),
        X_test_scaled.toarray(),
        vectorizer
    )


# =====================
# Loss Function
# =====================
def calculateNegativeLogLikelihood(X, y, weights, bias, l2=0.01):
  z = X @ weights + bias
  y_hat = 1/(1+np.exp(-z))

  eps = 1e-15
  y_hat = np.clip(y_hat, eps, 1-eps)

  # calculate negative log likelihood
  nll = -np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
  reg = (l2/2)*np.sum(weights ** 2)
  return nll + reg


# Logistic Regression Model
class CustomLogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000, l2 = 0.01):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = 0.0
        self.l2 = l2

    def sigmoid(self, z):
        z = np.clip(z, -20, 20)
        return 1 / (1 + np.exp(-z))
    
    def train(self, X, y, verbose = True):    
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # calcualte class weights
        classes, class_counts = np.unique(y, return_counts=True)
        class_weight = {c: n_samples/(len(classes) * count) for c, count in zip(classes, class_counts)}
        weights_vector = np.array([class_weight[label] for label in y])
        weights_sum = np.sum(weights_vector)

        # calculate gradients using the whole training set each iteration (change to stochastic for bigger dataset) 
        for i in range(self.num_iter):
            # multiply vectors X and weight
            # apply sigmoid to get prob
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # weighted gradient loss
            # w.r.t. weights
            # dw = (1/n_samples) * X.T @ ((y_pred - y) * weights_vector)
            dw = (X.T @ ((y_pred - y) * weights_vector)) / weights_sum

            #l2 regularization
            dw += self.l2 * self.weights

            # w.r.t. bias
            # db = (1/n_samples) * np.sum((y_pred - y) * weights_vector)
            db = np.sum((y_pred - y) * weights_vector) / weights_sum

            # gradient descent
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # calculate loss every 100 iterations
            if verbose and i % 100 == 0:
                loss = calculateNegativeLogLikelihood(X, y, self.weights, self.bias, self.l2)
                print(f"Iter {i}: loss={loss:.4f}")
    

    def predict(self, X, threshold = 0.4):
        z = X @ self.weights + self.bias
        y_pred_probs = self.sigmoid(z)
        return np.where(y_pred_probs >= threshold, 1, 0)


# ========================================
# Evaluation Metrics/Utilities
# ========================================
def crossValidate(X, y, lr, num_iter, l2, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    accuracy_scores = []

    for train_idx, test_idx in kf.split(X):
        # Train your custom model
        model = CustomLogisticRegression(lr, num_iter, l2)
        model.train(X[train_idx], y[train_idx], verbose=False)

        # Predict and evaluate
        y_pred = model.predict(X[test_idx])
        acc = np.mean(y_pred == y[test_idx])
        accuracy_scores.append(acc)

    return np.mean(accuracy_scores), np.std(accuracy_scores)


def printTopFeatures(model, vectorizer, k=15):
    features = np.array(vectorizer.get_feature_names_out())
    weights = model.weights

    top_red = np.argsort(weights)[-k:][::-1]
    top_green = np.argsort(weights)[:k]

    print("\n=========== Top Red Flag Features ===========")
    for f, w in zip(features[top_red], weights[top_red]):
        print(f"{f:<30}{w: .4f}")

    print("\n=========== Top Green Flag Features ===========")
    for f, w in zip(features[top_green], weights[top_green]):
        print(f"{f:<30}{w: .4f}")


# ========================================
# Main Experiment
# ========================================
def main():
    X_train, X_test, y_train, y_test = loadData()
    X_train_tfidf, X_test_tfidf, vectorizer = featureScaling(X_train, X_test)

    print("=========== Beginning Model Training ===========\n")
    model = CustomLogisticRegression(lr=0.01, num_iter=2000, l2=0.01)
    model.train(X_train_tfidf, y_train.values)

    # Predict 
    y_pred = model.predict(X_test_tfidf)   

    # Evaluation
    print("=========== Evaluation =========== \n")
    print(f"Shape of TF-IDF matrix: {X_train_tfidf.shape} \n")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)} \n")
    print(f"Classificaton report: \n {classification_report(y_test, y_pred)}")
    print(f"\nConfusion Matrix: \n {confusion_matrix(y_test, y_pred)} \n")

    X_all = np.vstack([X_train_tfidf, X_test_tfidf])
    y_all = np.hstack([y_train.values, y_test.values])

    mean_acc, std_acc = crossValidate(X_all, y_all, lr=0.01, num_iter=2000, l2=0.01)
    print(f"\nCross Validation Accuracy: {mean_acc: .3f} +- {std_acc: .3f}")

    printTopFeatures(model, vectorizer, TOP_K_FEATURES)

if __name__ == "__main__":
    main()

