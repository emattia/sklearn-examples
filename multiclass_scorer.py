from skopt.searchcv import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
def multiclass_confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    return {
        'micro_precision_score': precision_score(y, y_pred, average='micro'),
        'micro_recall_score': recall_score(y, y_pred, average='micro'),
        'micro_f1_score': f1_score(y, y_pred, average='micro'),
        'macro_precision_score': precision_score(y, y_pred, average='macro'),
        'macro_recall_score': recall_score(y, y_pred, average='macro'),
        'macro_f1_score': f1_score(y, y_pred, average='macro'),
        'weighted_precision_score': precision_score(y, y_pred, average='weighted'),
        'weighted_recall_score': recall_score(y, y_pred, average='weighted'),
        'weighted_f1_score': f1_score(y, y_pred, average='weighted'),
        'accuracy_score': accuracy_score(y, y_pred),
        'score': accuracy_score(y, y_pred)
    }

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)
svm = SVC()
svm.fit(X_train, y_train)
opt = BayesSearchCV(
    SVC(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'degree': (1, 8),  
        'kernel': ['linear', 'poly', 'rbf'],  
    },
    n_iter = 1,
    n_points = 2,
    cv = 5,
    scoring = multiclass_confusion_matrix_scorer, refit = 'score'
    # project_id='random'
)
opt.fit(X_train, y_train)
print("val. score: %s" % opt.best_score_)