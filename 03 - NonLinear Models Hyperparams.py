from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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

param_grid_poly = {
    'poly__degree': [2, 3, 4, 5]
}

poly_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('regressor', LinearRegression())
])

grid_search_poly = GridSearchCV(poly_model, param_grid=param_grid_poly, cv=5, scoring='neg_mean_squared_error')
grid_search_poly.fit(X_uncensored, y_uncensored)

print(f"\nMelhor grau polinomial: {grid_search_poly.best_params_['poly__degree']}")
print(f"Melhor MSE da validação cruzada: {-grid_search_poly.best_score_:.3f}")

param_grid_knn = {
    'regressor__n_neighbors': [3, 5, 7, 9]
}

knn_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', KNeighborsRegressor())
])

grid_search_knn = GridSearchCV(knn_model, param_grid=param_grid_knn, cv=5, scoring='neg_mean_squared_error')
grid_search_knn.fit(X_uncensored, y_uncensored)

print(f"\nMelhor valor de k para k-NN: {grid_search_knn.best_params_['regressor__n_neighbors']}")
print(f"Melhor MSE da validação cruzada para k-NN: {-grid_search_knn.best_score_:.3f}")

final_model = grid_search_knn.best_estimator_

final_model.fit(X_uncensored, y_uncensored)

final_predictions = final_model.predict(test_data.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']))

submission = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': final_predictions})
submission.to_csv('Nonlinear-submission-02.csv', index=False)

print("\nPrevisões salvas em 'nonlinear-model-submission.csv'.")
