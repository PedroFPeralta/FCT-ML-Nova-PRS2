from sklearn.model_selection import KFold


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
    regularization="ridge", lambda_=lambda_
)

print(f"Cross-Validated cMSE (mean): {mean_cMSE:.3f}")
print(f"Cross-Validated cMSE (std): {std_cMSE:.3f}")
