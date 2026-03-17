from pathlib import Path
import json

#use L2 regularization (ridge-style shrinkle)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

#load the data and use the TF-IDF representations from step 10
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path("hw03_12_baseline.py").resolve().parent/"train_supervised_classifier_week8"/"data"
with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.9
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

#train the classifier
#the model examines the TF-IDF features and it learns the weights that separate CORE from NEG

clf = LogisticRegression(
    max_iter=1000,
    n_jobs=1
)
clf.fit(X_train, y_train)

#test set predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

#evaluate the classifier with a confusion matrix that compares:
#true labels (y_test) 
#model predictions (y_pred)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

###ROC AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 3))

import numpy as np
nonzero_count = np.count_nonzero(clf.coef_[0])
print("Number of non-zero coefficients: ", nonzero_count)

#OUTPUT
# Confusion matrix:
# [[1906   75]
#  [ 131 1798]]

# Classification report:
#               precision    recall  f1-score   support

#            0       0.94      0.96      0.95      1981
#            1       0.96      0.93      0.95      1929

#     accuracy                           0.95      3910
#    macro avg       0.95      0.95      0.95      3910
# weighted avg       0.95      0.95      0.95      3910

# ROC AUC: 0.987
# Number of non-zero coefficients:  16198

from pathlib import Path
import joblib
MODEL_DIR = Path.cwd() / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer_L2.joblib")
joblib.dump(clf, MODEL_DIR / "merchant_logreg_L2.joblib")

print("Saved TF-IDF vectorizer and classifier to /models/")

words = np.array(vectorizer.get_feature_names_out())
coefficients = clf.coef_[0]

#sort by top 15 positive and top 15 negative scores
#print results
top_pos = np.argsort(coefficients) [-15:]
print("\nTop 15 positive-weighted words:")
for i in reversed(top_pos):
    print(f"{words[i]}: {coefficients[i]: .4f}")

top_neg = np.argsort(coefficients) [:15]
print("\nTop 15 negative-weighted words:")
for i in top_neg:
     print(f"{words[i]}: {coefficients[i]: .4f}")