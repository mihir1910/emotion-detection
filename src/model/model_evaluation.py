import pandas as pd
import numpy as np
import json
import pickle

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score

clf = pickle.load(open('model.pkl', 'rb'))
test_data = pd.read_csv('./data/features/test_bow.csv')

X_test=test_data.iloc[:,0:-1].values
y_test=test_data.iloc[:,-1].values

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

metrics = {
    'precision': precision,
    'recall': recall,
    'roc_auc': roc,
    'accuracy': accuracy
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)