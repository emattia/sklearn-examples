from skopt.searchcv import BayesSearchCV, SigOptSearchCV

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# log-uniform: understand as search over p = exp(x) by varying x
opt = SigOptSearchCV(
    SVC(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },
    n_iter = 1,
    n_points = 2,
    cv = 5,
    project_id = "random"
)

opt.fit(X_train, y_train)
print("val. score: %s" % opt.best_score_)
#print("test score: %s" % opt.score(X_test, y_test))