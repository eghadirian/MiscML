# feature selection
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile, SelectFromModel, RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

cancer = load_breast_cancer()
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test =train_test_split(X_w_noise, cancer.target, random_state=0, test_size=0.5)
# univariate feature selection
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train_selected, y_train)
print('score of univariate: {}'.format(lr.score(X_test_selected, y_test)))
# model based feature selection
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train_selected, y_train)
print('score od model based Random Forest: {}'.format(lr.score(X_test_selected, y_test)))
# RFE: Recursive Feature Elimination
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train_selected, y_train)
print('score od model based RFE: {}'.format(lr.score(X_test_selected, y_test)))