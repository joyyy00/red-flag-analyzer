import pandas as pd

# data labelings, counts, and empty rows
print("TEST 1 ==================================")

df = pd.read_csv("reddit_messages_cleaned.csv")
print("Columns:", df.columns.tolist())
print("Label value counts:\n", df["Label"].value_counts(dropna=False))
print("\nAny null Text?", df["CleanText"].isna().sum())
print("\nSample green_flag rows (5):\n", df[df["Label"]=="green_flag"]["CleanText"].head(5).to_list())
print("\nSample red_flag rows (5):\n", df[df["Label"]=="red_flag"]["CleanText"].head(5).to_list())
# check lengths
df["len_chars"] = df["CleanText"].astype(str).apply(len)
print("\nLength stats by label:\n", df.groupby("Label")["len_chars"].describe())


# check TFIDF 
print("TEST 2 ==================================")

from sklearn.feature_extraction.text import TfidfVectorizer
texts = df["CleanText"].astype(str)
vec = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X = vec.fit_transform(texts)
print("TF-IDF shape:", X.shape)
# number of zero rows
zero_rows = (X.sum(axis=1) == 0).A1.sum()
print("Rows with all-zero features:", zero_rows)
# fraction of zero rows by label
import numpy as np
labels = df["Label"].map({"green_flag":0,"red_flag":1}).values
zero_mask = (X.sum(axis=1) == 0).A1
print("Zero rows by label:\n", 
      pd.Series(labels[zero_mask]).value_counts())


# sklearn logistic regression w/class weight 
print("TEST 3 ==================================")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("sklearn LR report:\n", classification_report(y_test, y_pred))
print("Confusion:\n", confusion_matrix(y_test, y_pred))

# predicted probability distribution
print("TEST 4 ==================================")
# for sklearn
probs = clf.predict_proba(X_test)[:,1]    # prob of class 1
import numpy as np
print("sklearn prob summary (class1):", np.percentile(probs, [0,25,50,75,100]))
# for your custom model: you will need to compute probabilities similarly
# after training your model: probs_custom = model.sigmoid(X_test @ weights + bias)


# top features for each class
print("TEST 5 ==================================")
feat_names = vec.get_feature_names_out()
coefs = clf.coef_[0]
top_pos = sorted(zip(coefs, feat_names), reverse=True)[:20]
top_neg = sorted(zip(coefs, feat_names), reverse=False)[:20]
print("Top features for predicting RED (positive):", [t for _,t in top_pos[:20]])
print("Top features for predicting GREEN (negative):", [t for _,t in top_neg[:20]])
