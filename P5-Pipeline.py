from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV

# pipeline
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
pipe = Pipeline([('Scaler', MinMaxScaler()),('svm', SVC())])
pipe.fit(X_train, y_train)
print('score: {}'.format(pipe.score(X_test, y_test)))
# gridsearch+pipeline
param_grid = {'svm__C':[10**(i-3) for i in range(6)], 'svm__gamma':[10**(i-3) for i in range(6)]}
grid = GridSearchCV(pipe, param_grid, cv = 5)
grid.fit(X_train, y_train)
print('Score: {}'.format(grid.score(X_test, y_test)))
# make pipeline
pipe = make_pipeline(MinMaxScaler(), SVC(C=100))