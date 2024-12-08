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

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_test_scaled = scaler_X.transform(test_data.drop(columns=['id']))

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

def cMSE_loss(y, y_hat, c):
    err = y - y_hat
    loss = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(loss) / loss.shape[0]

def cMSE_derivative(X, y, y_hat, c):
    err = y - y_hat
    grad = -2 * (1 - c)[:, None] * (err[:, None] * X)
    grad += -2 * c[:, None] * np.maximum(0, err)[:, None] * X
    return np.mean(grad, axis=0)

def gradient_descent(X, y, c, alpha=0.001, n_iter=5000, regularization=None, lambda_=0.1):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []

    for i in range(n_iter):
        y_hat = X @ theta
        loss = cMSE_loss(y, y_hat, c)
        losses.append(loss)

        grad = cMSE_derivative(X, y, y_hat, c)

        if regularization == "lasso":
            grad += lambda_ * np.sign(theta)
        elif regularization == "ridge":
            grad += 2 * lambda_ * theta

        theta -= alpha * grad

        if i % 500 == 0:  # Log a cada 500 iterações
            print(f"Iteração {i}, cMSE Loss: {loss:.4f}, theta: {theta}")

    return theta, losses

# Configurações
alpha = 0.0001
n_iter = 10000
lambda_ = 0.1

# Treinar modelos com diferentes regularizações
theta, losses = gradient_descent(X_scaled, y_scaled, c.values, alpha=alpha, n_iter=n_iter)
theta_lasso, losses_lasso = gradient_descent(X_scaled, y_scaled, c.values, alpha=alpha, n_iter=n_iter, regularization="lasso", lambda_=lambda_)
theta_ridge, losses_ridge = gradient_descent(X_scaled, y_scaled, c.values, alpha=alpha, n_iter=n_iter, regularization="ridge", lambda_=lambda_)

# Avaliar os modelos no conjunto de treino
y_train_pred_scaled = X_scaled @ theta
y_train_pred_scaled_lasso = X_scaled @ theta_lasso
y_train_pred_scaled_ridge = X_scaled @ theta_ridge

# Desnormalizar previsões de treino
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
y_train_pred_lasso = scaler_y.inverse_transform(y_train_pred_scaled_lasso.reshape(-1, 1)).flatten()
y_train_pred_ridge = scaler_y.inverse_transform(y_train_pred_scaled_ridge.reshape(-1, 1)).flatten()

# Calcular cMSE no conjunto de treino
cMSE_train = cMSE_loss(y.values, y_train_pred, c.values)
cMSE_train_lasso = cMSE_loss(y.values, y_train_pred_lasso, c.values)
cMSE_train_ridge = cMSE_loss(y.values, y_train_pred_ridge, c.values)

print("Desempenho no conjunto de treino:")
print(f"No Regularization cMSE: {cMSE_train:.3f}")
print(f"Lasso Regularization cMSE: {cMSE_train_lasso:.3f}")
print(f"Ridge Regularization cMSE: {cMSE_train_ridge:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(losses, label='No Regularization')
plt.plot(losses_lasso, label='Lasso Regularization')
plt.plot(losses_ridge, label='Ridge Regularization')
plt.xlabel('Iteration')
plt.ylabel('Loss (cMSE)')
plt.legend()
plt.title('Loss over Iterations')
plt.show()

y_test_pred_scaled = X_test_scaled @ theta
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

print("Estatísticas de previsões de teste:")
print(f"Média: {np.mean(y_test_pred):.2f}, Máximo: {np.max(y_test_pred):.2f}, Mínimo: {np.min(y_test_pred):.2f}")

submission = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': y_test_pred})
submission.to_csv('cMSE-baseline-submission-01.csv', index=False)
