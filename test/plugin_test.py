from tools.PlugIn import PlugInRule
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
X_ood, y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)


def test_doc_examples():
    """
    Test it works
    """
    X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)

    clf = PlugInRule(model=LogisticRegression())
    clf.fit(X_tr, y_tr)
    assert sum(clf.predict(X_te)) == 830


def test_quantile_works():
    """
    Test that if you increase the parameter of coverage, the number of
    instance accepted increases
    """
    det = PlugInRule(model=LogisticRegression())
    det.fit(X_tr, y_tr)
    # Compute quantiles
    cov99 = sum(det.predict(X_te, cov=0.99))
    cov9 = sum(det.predict(X_te, cov=0.9))
    cov5 = sum(det.predict(X_te, cov=0.5))
    assert cov99 > cov9 > cov5


def test_plugin_fitted():
    """
    Check that no NaNs are present in the shap values.
    """
    # clf = PlugInRule(model=XGBClassifier())
    # with pytest.raises(ValueError):
    # clf.predict(X_te)


def test_thetas_estimated():
    """
    Check that theta has being called

    """

    clf = PlugInRule(model=LogisticRegression())
    clf.fit(X_tr, y_tr)
    clf.predict(X_te)
    assert clf.theta is not None
