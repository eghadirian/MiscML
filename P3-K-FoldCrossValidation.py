from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

iris = load_iris()
logreg = LogisticRegression()
score = cross_val_score(logreg, iris.data, iris.target, cv=5)
print('Average cross validation score: {}'.format(score.mean()))

# grid search + cross validation
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
best_score = 0.
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        score = scores.mean()
        if score > best_score:
            best_score = score
            best_parameter ={'gamma':gamma, 'C':C}
svm = SVC(**best_parameter)
print('Score on test data: {}'.format(svm.fit(X_trainval, y_trainval).score(X_test, y_test)))

# GridSearchCV
param_grid={'gamma':[0.001, 0.01, 0.1, 1, 10, 100], 'C':[0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
print('Score on test data: {}'.format(grid_search.fit(X_trainval, y_trainval).score(X_test, y_test)))
