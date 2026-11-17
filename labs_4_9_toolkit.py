import numpy as np
import pandas as pd


def train_test_split_scratch(X, y, test_frac, seed):
    X, y = np.asarray(X, float), np.asarray(y)
    idx = np.random.default_rng(seed).permutation(len(y))
    n_test = int(len(y) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def fit_simple_lr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    denominator = np.sum(xc ** 2)
    if denominator == 0:
        raise ValueError("The single feature must have nonzero variance.")
    w = np.sum(xc * yc) / denominator
    return float(w), float(y.mean() - w * x.mean())


def design_matrix(X):
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("Pass a two-dimensional feature matrix.")
    return np.c_[np.ones(len(X)), X]


def fit_normal_equation(X, y):
    A = design_matrix(X)
    return np.linalg.solve(A.T @ A, A.T @ np.asarray(y, float))


def fit_gradient_descent(X, y, lr, n_iters):
    A, y = design_matrix(X), np.asarray(y, float)
    theta, losses = np.zeros(A.shape[1]), []
    for _ in range(n_iters):
        theta -= lr * (2 / len(y)) * (A.T @ (A @ theta - y))
        losses.append(float(np.mean((A @ theta - y) ** 2)))
    return theta, losses


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score_scratch(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    if sst == 0:
        raise ValueError("R-squared is undefined for a constant target.")
    return float(1 - np.sum((y_true - y_pred) ** 2) / sst)


def standardise(X_train, X_test):
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def original_units(theta, mu, sigma):
    w = np.asarray(theta)[1:] / sigma
    return np.r_[theta[0] - np.dot(w, mu), w]


def fit_ridge(X, y, lam):
    A = design_matrix(X)
    penalty = np.eye(A.shape[1])
    penalty[0, 0] = 0
    return np.linalg.solve(A.T @ A + lam * penalty, A.T @ y)


def _fold_indices(n, k, seed):
    if not 2 <= k <= n:
        raise ValueError("The number of folds must be between 2 and n.")
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def k_fold_cv(X, y, lam, k=5, seed=0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    folds, scores = _fold_indices(len(y), k, seed), []
    for i, va in enumerate(folds):
        tr = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        theta = fit_ridge(X[tr], y[tr], lam)
        scores.append(rmse(y[va], design_matrix(X[va]) @ theta))
    return float(np.mean(scores))


def k_fold_cv_ridge_fold_scaled(X, y, lam, k=5, seed=0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    folds, scores = _fold_indices(len(y), k, seed), []
    for i, va in enumerate(folds):
        tr = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        Z_tr, Z_va, _, _ = standardise(X[tr], X[va])
        theta = fit_ridge(Z_tr, y[tr], lam)
        scores.append(rmse(y[va], design_matrix(Z_va) @ theta))
    return float(np.mean(scores))


def soft_threshold(v, t):
    v = np.asarray(v, float)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0)


def fit_lasso(X, y, lam, lr=0.05, n_iters=30000):
    A, y = design_matrix(X), np.asarray(y, float)
    theta = np.zeros(A.shape[1])
    for _ in range(n_iters):
        theta -= lr * (2 / len(y)) * (A.T @ (A @ theta - y))
        theta[1:] = soft_threshold(theta[1:], lr * lam)
    return theta


def pairwise_distances(X_query, X_train):
    X_query, X_train = np.asarray(X_query, float), np.asarray(X_train, float)
    squared = (np.sum(X_query ** 2, axis=1)[:, None]
               + np.sum(X_train ** 2, axis=1)[None, :]
               - 2 * X_query @ X_train.T)
    return np.sqrt(np.maximum(squared, 0))


def _knn_neighbours(X_train, y_train, X_query, k):
    if not 1 <= k <= len(y_train):
        raise ValueError("k must be between 1 and the training-set size.")
    distances = pairwise_distances(X_query, X_train)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return distances, indices, np.asarray(y_train)[indices]


def knn_predict(X_train, y_train, X_query, k):
    _, _, neighbour_labels = _knn_neighbours(X_train, y_train, X_query, k)
    return (neighbour_labels.mean(axis=1) > 0.5).astype(int)


def knn_predict_weighted(X_train, y_train, X_query, k):
    distances, indices, neighbour_labels = _knn_neighbours(X_train, y_train, X_query, k)
    weights = 1 / (np.take_along_axis(distances, indices, axis=1) + 1e-12)
    probability = np.sum(weights * neighbour_labels, axis=1) / weights.sum(axis=1)
    return (probability > 0.5).astype(int)


def k_fold_cv_knn(X, y, k, folds=5, seed=0):
    X, y = np.asarray(X, float), np.asarray(y, int)
    groups, scores = _fold_indices(len(y), folds, seed), []
    for i, va in enumerate(groups):
        tr = np.concatenate([fold for j, fold in enumerate(groups) if j != i])
        scores.append(np.mean(knn_predict(X[tr], y[tr], X[va], k) == y[va]))
    return float(np.mean(scores))


def k_fold_cv_knn_fold_scaled(X, y, k, folds=5, seed=0):
    X, y = np.asarray(X, float), np.asarray(y, int)
    groups, scores = _fold_indices(len(y), folds, seed), []
    for i, va in enumerate(groups):
        tr = np.concatenate([fold for j, fold in enumerate(groups) if j != i])
        Z_tr, Z_va, _, _ = standardise(X[tr], X[va])
        scores.append(np.mean(knn_predict(Z_tr, y[tr], Z_va, k) == y[va]))
    return float(np.mean(scores))


def confusion_matrix_scratch(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    return np.array([[np.sum((y_true == 0) & (y_pred == 0)),
                      np.sum((y_true == 0) & (y_pred == 1))],
                     [np.sum((y_true == 1) & (y_pred == 0)),
                      np.sum((y_true == 1) & (y_pred == 1))]])


def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_scratch(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"accuracy": float((tp + tn) / (tn + fp + fn + tp)),
            "precision": float(precision), "recall": float(recall), "F1": float(f1)}


def sigmoid(z):
    z = np.asarray(z, float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def log_likelihood(X, y, theta):
    z = design_matrix(X) @ theta
    return float(np.sum(np.asarray(y) * z - np.logaddexp(0, z)))


def fit_logistic(X, y, lr=0.1, n_iters=2000):
    A, y = design_matrix(X), np.asarray(y, float)
    theta, history = np.zeros(A.shape[1]), []
    for _ in range(n_iters):
        theta += (lr / len(y)) * (A.T @ (y - sigmoid(A @ theta)))
        z = A @ theta
        history.append(float(np.sum(y * z - np.logaddexp(0, z))))
    return theta, history


def roc_curve_scratch(y_true, p):
    y_true, p = np.asarray(y_true, int), np.asarray(p, float)
    positives, negatives = np.sum(y_true == 1), np.sum(y_true == 0)
    if not positives or not negatives:
        raise ValueError("ROC needs both classes in the evaluation set.")
    if len(y_true) != len(p) or not np.all(np.isfinite(p)):
        raise ValueError("Supply one finite probability per label.")
    order = np.argsort(-p, kind="stable")
    labels, scores = y_true[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(labels == 1)[ends]
    fp = ends + 1 - tp
    return np.r_[0.0, fp / negatives], np.r_[0.0, tp / positives]


def auc_trapezoid(fpr, tpr):
    fpr, tpr = np.asarray(fpr, float), np.asarray(tpr, float)
    if np.any(np.diff(fpr) < 0):
        raise ValueError("FPR must be in nondecreasing order.")
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def gaussian_pdf(x, mu, var):
    var = np.maximum(np.asarray(var, float), 1e-12)
    return np.exp(-0.5 * (np.asarray(x) - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def fit_gaussian_nb(X, y):
    X, y = np.asarray(X, float), np.asarray(y, int)
    priors, means, variances = [], [], []
    for c in (0, 1):
        Xc = X[y == c]
        if not len(Xc):
            raise ValueError("Both classes must occur in the training data.")
        priors.append(len(Xc) / len(X))
        means.append(Xc.mean(axis=0))
        variances.append(Xc.var(axis=0, ddof=0))
    return np.array(priors), np.array(means), np.array(variances)


def _gaussian_log_scores(X, priors, means, variances):
    X = np.asarray(X, float)
    variances = np.maximum(np.asarray(variances, float), 1e-12)
    return (np.log(priors)[None, :]
            - 0.5 * np.sum(np.log(2 * np.pi * variances)[None, :, :]
                           + (X[:, None, :] - means[None, :, :]) ** 2
                           / variances[None, :, :], axis=2))


def predict_proba_nb(X, priors, means, variances):
    scores = _gaussian_log_scores(X, priors, means, variances)
    return sigmoid(scores[:, 1] - scores[:, 0])


def fit_categorical_nb(e, y, l):
    e, y = np.asarray(e), np.asarray(y, int)
    categories = np.unique(e)
    if len(categories) != 4:
        raise ValueError("Declare the known four-category vocabulary if training omits a category.")
    probabilities = np.zeros((2, 4))
    for c in (0, 1):
        values = e[y == c]
        probabilities[c] = (np.array([np.sum(values == j) for j in categories]) + l) / (len(values) + l * 4)
    return pd.DataFrame(probabilities, index=[0, 1], columns=categories)


def predict_proba_mixed_nb(X, e, priors, means, variances, category_table):
    scores = _gaussian_log_scores(X, priors, means, variances)
    with np.errstate(divide="ignore"):
        for c in (0, 1):
            scores[:, c] += np.log(category_table.loc[c, list(e)].to_numpy(float))
    return sigmoid(scores[:, 1] - scores[:, 0])


def random_init(X, K, seed):
    X = np.asarray(X, float)
    idx = np.random.default_rng(seed).choice(len(X), K, replace=False)
    return X[idx].copy()


def assign_clusters(X, centroids):
    return np.argmin(pairwise_distances(X, centroids), axis=1)


def update_centroids(X, labels, centroids):
    X, labels = np.asarray(X, float), np.asarray(labels, int)
    updated = np.asarray(centroids, float).copy()
    for c in range(len(centroids)):
        points = X[labels == c]
        if len(points):
            updated[c] = points.mean(axis=0)
    return updated


def distortion(X, labels, centroids):
    return float(np.sum((np.asarray(X) - np.asarray(centroids)[labels]) ** 2))


def kmeans(X, init_centroids, max_iter=100):
    X, centroids = np.asarray(X, float), np.asarray(init_centroids, float).copy()
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    for iteration in range(1, max_iter + 1):
        labels = assign_clusters(X, centroids)
        updated = update_centroids(X, labels, centroids)
        converged = np.allclose(updated, centroids)
        centroids = updated
        if converged:
            break
    labels = assign_clusters(X, centroids)
    return centroids, labels, distortion(X, labels, centroids), iteration


def best_of_kmeans(X, K, n_init=10):
    if n_init < 1:
        raise ValueError("n_init must be positive.")
    best = None
    for seed in range(n_init):
        result = kmeans(X, random_init(X, K, seed))
        if best is None or result[2] < best[2]:
            best = result
    return best


def kmeans_pp_init(X, K, seed):
    X = np.asarray(X, float)
    if not 1 <= K <= len(X):
        raise ValueError("K must be between 1 and the number of data rows.")
    rng = np.random.default_rng(seed)
    chosen = [int(rng.choice(len(X)))]
    while len(chosen) < K:
        nearest_squared = pairwise_distances(X, X[chosen]).min(axis=1) ** 2
        nearest_squared[chosen] = 0
        if nearest_squared.sum() == 0:
            remaining = np.setdiff1d(np.arange(len(X)), chosen)
            chosen.append(int(rng.choice(remaining)))
        else:
            chosen.append(int(rng.choice(len(X), p=nearest_squared / nearest_squared.sum())))
    return X[chosen].copy()
