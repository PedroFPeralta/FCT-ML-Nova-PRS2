from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_path = "train_data.csv"
test_path = "test_data.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data_clean = train_data.dropna(subset=['SurvivalTime'])

train_data_clean = train_data_clean.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'])
test_data = test_data.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'])

X = train_data_clean.drop(columns=['SurvivalTime'])
y = train_data_clean['SurvivalTime']
c = train_data_clean['Censored']
X = X.drop(columns=['id', 'Censored'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_data.drop(columns=['id']))


def cMSE_loss(y, y_hat, c):
    err = y - y_hat
    loss = (1 - c) * err ** 2 + c * np.maximum(0, err) ** 2
    return np.sum(loss) / loss.shape[0]


def cMSE_derivative(X, y, y_hat, c):
    err = y - y_hat
    grad = -2 * (1 - c)[:, None] * (err[:, None] * X)
    grad += -2 * c[:, None] * np.maximum(0, err)[:, None] * X
    return np.mean(grad, axis=0)


def gradient_descent(X, y, c, alpha=0.01, n_iter=1000, regularization=None, lambda_=0.1):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []

    for _ in range(n_iter):
        y_hat = X @ theta
        loss = cMSE_loss(y, y_hat, c)
        losses.append(loss)

        grad = cMSE_derivative(X, y, y_hat, c)

        if regularization == "lasso":
            grad += lambda_ * np.sign(theta)
        elif regularization == "ridge":
            grad += 2 * lambda_ * theta

        theta -= alpha * grad

    return theta, losses


# Validação Cruzada com K-Fold
def cross_validate_cMSE(X, y, c, alpha=0.01, n_iter=1000, k=5, regularization=None, lambda_=0.1):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cMSE_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        c_train, c_val = c[train_index], c[val_index]

        theta, _ = gradient_descent(
            X_train, y_train, c_train,
            alpha=alpha, n_iter=n_iter,
            regularization=regularization, lambda_=lambda_
        )

        y_val_pred = X_val @ theta

        val_loss = cMSE_loss(y_val, y_val_pred, c_val)
        cMSE_scores.append(val_loss)

    return np.mean(cMSE_scores), np.std(cMSE_scores)


# Configurações
alpha = 0.01
n_iter = 1000
k = 5  # Número de folds
lambda_ = 0.1

mean_cMSE, std_cMSE = cross_validate_cMSE(
    X_scaled, y.values, c.values,
    alpha=alpha, n_iter=n_iter, k=k,
    regularization='ridge', lambda_=lambda_
)

print(f"Cross-Validated cMSE (mean): {mean_cMSE:.3f}")
print(f"Cross-Validated cMSE (std): {std_cMSE:.3f}")

theta, _ = gradient_descent(X_scaled, y.values, c.values, alpha=alpha, n_iter=n_iter, regularization="ridge",
                            lambda_=lambda_)

y_test_pred = X_test_scaled @ theta

submission = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': y_test_pred})
submission.to_csv('cMSE-baseline-submission-02.csv', index=False)
