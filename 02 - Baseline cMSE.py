from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
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
    loss = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(loss)/loss.shape[0]

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

alpha = 0.01
n_iter = 1000
lambda_ = 0.1

theta, losses = gradient_descent(X_scaled, y.values, c.values, alpha=alpha, n_iter=n_iter)
# Com Lasso
theta_lasso, losses_lasso = gradient_descent(X_scaled, y.values, c.values, alpha=alpha, n_iter=n_iter, regularization="lasso", lambda_=lambda_)
# Com Ridge
theta_ridge, losses_ridge = gradient_descent(X_scaled, y.values, c.values, alpha=alpha, n_iter=n_iter, regularization="ridge", lambda_=lambda_)

# Plotar as perdas
plt.figure(figsize=(8, 6))
plt.plot(losses, label='No Regularization')
plt.plot(losses_lasso, label='Lasso Regularization')
plt.plot(losses_ridge, label='Ridge Regularization')
plt.xlabel('Iteration')
plt.ylabel('Loss (cMSE)')
plt.legend()
plt.title('Loss over Iterations')
plt.show()

y_test_pred = X_test_scaled @ theta

submission = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': y_test_pred})
submission.to_csv('cMSE-baseline-submission-02.csv', index=False)
