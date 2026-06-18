"""King–Zeng (2001) Rare Events Logistic：参数修正与概率修正（向量化，无 n×n 矩阵）。"""

from __future__ import annotations

import numpy as np
from numpy.linalg import pinv
from sklearn.linear_model import LogisticRegression


def _design_matrix(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def kz_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    ridge: float = 1e-6,
    lr_kwargs: dict | None = None,
) -> np.ndarray:
    kw = dict(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, class_weight=None)
    if lr_kwargs:
        kw.update(lr_kwargs)

    clf = LogisticRegression(**kw)
    clf.fit(X_train, y_train)

    X_tr = _design_matrix(X_train)
    X_pr = _design_matrix(X_pred)
    beta = np.concatenate([[clf.intercept_[0]], clf.coef_.ravel()])

    pi = clf.predict_proba(X_train)[:, 1]
    w = pi * (1.0 - pi)
    xi = (0.5 - pi) * pi * (1.0 - pi)

    # (X' W X) beta = X' (w * col)  — 避免构造 n×n 的 W
    XtWX = X_tr.T @ (w[:, None] * X_tr) + ridge * np.eye(X_tr.shape[1])
    bias = pinv(XtWX) @ (X_tr.T @ (w * xi))
    beta_tilde = beta - bias

    pi_pred = 1.0 / (1.0 + np.exp(-X_pr @ beta_tilde))
    V = pinv(XtWX)
    quad = np.sum((X_pr @ V) * X_pr, axis=1)
    p_tilde = pi_pred + (0.5 - pi_pred) * pi_pred * (1.0 - pi_pred) * quad
    return np.clip(p_tilde, 0.0, 1.0)
