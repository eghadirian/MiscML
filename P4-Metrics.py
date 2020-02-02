from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_recall_curve, \
    average_precision_score, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
y = digits.target == 9
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)
# imbalanced data set
Dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
print('Score, most frequent: {}'.format(Dummy_majority.score(X_test, y_test)))
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
print('score, decision tree: {}'.format(tree.score(X_test, y_test)))
Dummy = DummyClassifier().fit(X_train, y_train)
print('Score, dummy: {}'.format(Dummy.score(X_test, y_test)))
lr = LogisticRegression(C=0.1).fit(X_train, y_train)
print('Score, logistic regression: {}'.format(lr.score(X_test, y_test)))
# confusion matrix
pred_lr = lr.predict(X_test)
confusion = confusion_matrix(y_test, pred_lr)
print('Confusion Matrix for LogisticRegression:\n{}'.format(confusion))
print('F1-score for LogisticRegression: {}'.format(f1_score(y_test, pred_lr)))
# for multiclass f1-score can be used with average='micro','macro' or 'weighted'
print('Report:\n{}'.format(classification_report(y_test, pred_lr)))
# higher threshold higher precision, lower recall
precision, recall, threshold = precision_recall_curve(y_test, lr.decision_function(X_test))
close_zero = np.argmin(np.abs(threshold))
plt.plot(precision[close_zero], recall[close_zero], 'o')
plt.plot(precision, recall, '-')
plt.show()
print('Average precision score: {}'.format(average_precision_score(y_test,lr.decision_function(X_test))))
# ROC(Receiver Operating Characteristics) False Positive Rate (FPR) vs True Positive Rate(TPR: recall)
fpr, tpr, threshold = roc_curve(y_test, lr.decision_function(X_test))
plt.plot(fpr, tpr, label='ROC curve')
close_zero = np.argmin(np.abs(threshold))
plt.plot(fpr[close_zero], tpr[close_zero], 'o')
plt.show()
# AUC (area under curve)
print('AUC: {}'.format(roc_auc_score(y_test, lr.decision_function(X_test))))
# AUC is very useful on imbalanced data, but we'll need to adjust threshold. We get information that
# cannot be found from accuracy score only