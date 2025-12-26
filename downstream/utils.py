import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

def preprocess_label(label, task="regression"):
    label = np.asarray(label)

    if task == "classification":
        if label.ndim > 1:
            if label.shape[1] == 1:
                label = label.squeeze()
            else:
                label = np.argmax(label, axis=1)

        if np.issubdtype(label.dtype, np.floating):
            uniq = np.unique(label)
            mapping = {v: i for i, v in enumerate(uniq)}
            label = np.vectorize(mapping.get)(label).astype(int)
        else:
            label = label.astype(int)

    elif task == "regression":
        label = label.astype(float)
        if label.ndim == 2 and label.shape[1] == 1:
            label = label.squeeze(1)
        # [N, D] (D>1) 保持不变

    else:
        raise ValueError(f"Unknown task type: {task}")

    return label


def regression(fea, label, mask, svr_params=None):
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

    models = []
    y_pred = None
    metrics = {}

    # ===== 单维回归 =====
    if y_train.ndim == 1:
        model = SVR(**svr_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        models = model
        metrics = {
            "mse": mean_squared_error(y_val, y_pred),
            "r2": r2_score(y_val, y_pred)
        }

    # ===== 多维回归：每个维度一个回归器 =====
    else:
        n_dim = y_train.shape[1]
        y_pred = np.zeros((X_val.shape[0], n_dim))
        metrics["per_dim"] = []

        for d in range(n_dim):
            model = SVR(**svr_params)
            model.fit(X_train, y_train[:, d])
            y_pred[:, d] = model.predict(X_val)

            dim_metrics = {
                "mse": mean_squared_error(y_val[:, d], y_pred[:, d]),
                "r2": r2_score(y_val[:, d], y_pred[:, d])
            }
            metrics["per_dim"].append(dim_metrics)
            models.append(model)

    return models, y_pred, metrics


def classification(fea, label, mask, svm_params=None):
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
