import numpy as np

def preprocess_label(label, task="regression"):
    """
    task: 'regression' or 'classification'
    """
    label = np.asarray(label)

    # 如果是 [N, 1] 或 one-hot
    if label.ndim > 1:
        if label.shape[1] == 1:
            label = label.squeeze()
        else:
            # one-hot → class index
            label = np.argmax(label, axis=1)

    if task == "classification":
        if np.issubdtype(label.dtype, np.floating):
            # float → 离散化
            uniq = np.unique(label)
            mapping = {v: i for i, v in enumerate(uniq)}
            label = np.vectorize(mapping.get)(label).astype(int)
        else:
            label = label.astype(int)

    elif task == "regression":
        label = label.astype(float)

    return label

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def svr_regression(fea, label, mask, svr_params=None):
    """
    SVR regression with automatic label handling
    """
    fea = np.asarray(fea)
    mask = np.asarray(mask).astype(bool)

    label = preprocess_label(label, task="regression")

    X_train, X_val = fea[mask], fea[~mask]
    y_train, y_val = label[mask], label[~mask]

    if svr_params is None:
        svr_params = dict(
            kernel="rbf",
            C=1.0,
            epsilon=0.1
        )

    model = SVR(**svr_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    metrics = {
        "mse": mean_squared_error(y_val, y_pred),
        "r2": r2_score(y_val, y_pred)
    }

    return model, y_pred, metrics

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def svm_classification(fea, label, mask, svm_params=None):
    """
    SVM classification with automatic label handling
    """
    fea = np.asarray(fea)
    mask = np.asarray(mask).astype(bool)

    label = preprocess_label(label, task="classification")

    X_train, X_val = fea[mask], fea[~mask]
    y_train, y_val = label[mask], label[~mask]

    if svm_params is None:
        svm_params = dict(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )

    model = SVC(**svm_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "report": classification_report(y_val, y_pred, zero_division=0)
    }

    return model, y_pred, metrics
