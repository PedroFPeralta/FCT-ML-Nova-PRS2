from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_path = "train_data.csv"
test_path = "test_data.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data_clean = train_data.dropna(subset=['SurvivalTime'])

train_data_clean = train_data_clean.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'])

X_uncensored = train_data_clean.drop(columns=['SurvivalTime', 'Censored'])
y_uncensored = train_data_clean['SurvivalTime']

def train_and_evaluate(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f'Mean CV MSE: {-cv_scores.mean():.3f}')
    return cv_scores.mean()

linear_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

print("\nAvaliação do Modelo Linear:")
train_and_evaluate(linear_model, X_uncensored, y_uncensored)

degree_range = [2, 3, 4, 5]
best_degree = 2

for degree in degree_range:
    poly_model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('regressor', LinearRegression())
    ])
    print(f"\nAvaliação do Modelo Polinomial (Grau {degree}):")
    mean_cv_mse = train_and_evaluate(poly_model, X_uncensored, y_uncensored)
    if degree == 2 or mean_cv_mse < best_degree:
        best_degree = degree

print(f"\nMelhor grau do polinômio: {best_degree}")

best_poly_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
    ('regressor', LinearRegression())
])

print("\nAvaliação do Melhor Modelo Polinomial:")
train_and_evaluate(best_poly_model, X_uncensored, y_uncensored)

k_range = [3, 5, 7, 9]
best_k = 3  # melhor k (k=3)

for k in k_range:
    knn_model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', KNeighborsRegressor(n_neighbors=k))
    ])
    print(f"\nAvaliação do Modelo k-NN (k={k}):")
    mean_cv_mse = train_and_evaluate(knn_model, X_uncensored, y_uncensored)
    if k == 3 or mean_cv_mse < best_k:
        best_k = k

print(f"\nMelhor valor de k para k-NN: {best_k}")

best_knn_model = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Normalização das features
    ('regressor', KNeighborsRegressor(n_neighbors=best_k))
])

print("\nAvaliação do Melhor Modelo k-NN:")
train_and_evaluate(best_knn_model, X_uncensored, y_uncensored)

final_model = best_poly_model

final_model.fit(X_uncensored, y_uncensored)

final_predictions = final_model.predict(test_data.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']))

submission = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': final_predictions})
submission.to_csv('Nonlinear-pol-submission-01.csv', index=False)

final_model = best_knn_model

final_model.fit(X_uncensored, y_uncensored)

final_predictions = final_model.predict(test_data.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']))

submission = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': final_predictions})
submission.to_csv('Nonlinear-knn-submission-01.csv', index=False)
