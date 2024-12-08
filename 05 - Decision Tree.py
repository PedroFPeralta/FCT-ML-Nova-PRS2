from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_path = "train_data.csv"
test_path = "test_data.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data_clean = train_data.dropna(subset=['SurvivalTime'])

features_with_missing = train_data_clean.columns.difference(['SurvivalTime', 'Censored'])

def decision_tree_imputation(df, target_column, features):
    imputed_df = df.copy()
    for feature in features:
        not_missing = imputed_df[~imputed_df[feature].isnull()]
        missing = imputed_df[imputed_df[feature].isnull()]

        if not missing.empty:
            tree = DecisionTreeRegressor(random_state=42)
            tree.fit(not_missing[features_with_missing.drop(feature)], not_missing[feature])

            imputed_df.loc[missing.index, feature] = tree.predict(missing[features_with_missing.drop(feature)])

    return imputed_df

train_data_clean = decision_tree_imputation(train_data_clean, 'SurvivalTime', features_with_missing)
test_data_imputed = decision_tree_imputation(test_data, 'SurvivalTime', features_with_missing)

X_uncensored = train_data_clean.drop(columns=['SurvivalTime', 'Censored'])
y_uncensored = train_data_clean['SurvivalTime']

def train_and_evaluate(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Mean CV MSE: {-cv_scores.mean():.3f}")
    return cv_scores.mean()

linear_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

print("\nLinear Model Evaluation:")
train_and_evaluate(linear_model, X_uncensored, y_uncensored)

degree_range = [2, 3, 4, 5]
best_degree = 2
best_poly_model = None
lowest_cv_mse = float('-inf')

for degree in degree_range:
    poly_model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('regressor', LinearRegression())
    ])
    print(f"\nEvaluating Polynomial Model (Degree {degree}):")
    cv_mse = train_and_evaluate(poly_model, X_uncensored, y_uncensored)

    if cv_mse > lowest_cv_mse:
        lowest_cv_mse = cv_mse
        best_degree = degree
        best_poly_model = poly_model

print(f"\nBest Polynomial Degree: {best_degree}")

k_range = [3, 5, 7, 9]
best_k = 3
best_knn_model = None
lowest_cv_mse_knn = float('-inf')

for k in k_range:
    knn_model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', KNeighborsRegressor(n_neighbors=k))
    ])
    print(f"\nEvaluating k-NN Model (k={k}):")
    cv_mse = train_and_evaluate(knn_model, X_uncensored, y_uncensored)
    if cv_mse > lowest_cv_mse_knn:
        lowest_cv_mse_knn = cv_mse
        best_k = k
        best_knn_model = knn_model

print(f"\nBest k for k-NN: {best_k}")

X_test = test_data_imputed

print("\nTraining Final Polynomial Model...")
best_poly_model.fit(X_uncensored, y_uncensored)

train_and_evaluate(best_poly_model, X_uncensored, y_uncensored)

final_predictions = best_poly_model.predict(X_test)

train_predictions_poly = best_poly_model.predict(X_uncensored)
plt.figure(figsize=(8, 6))
plt.scatter(y_uncensored, train_predictions_poly, alpha=0.5)
plt.plot([y_uncensored.min(), y_uncensored.max()], [y_uncensored.min(), y_uncensored.max()], 'k--', lw=2)
plt.xlabel('Survival Time (y)')
plt.ylabel('Predicted Survival Time (y_hat)')
plt.title('y vs. y_hat Plot - Polynomial Model')
plt.show()

submission_poly = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': final_predictions})
submission_poly.to_csv('Polynomial_Model_Submission.csv', index=False)

print("\nTraining Final k-NN Model...")
best_knn_model.fit(X_uncensored, y_uncensored)

train_and_evaluate(best_knn_model, X_uncensored, y_uncensored)

final_predictions_knn = best_knn_model.predict(X_test)

train_predictions_knn = best_knn_model.predict(X_uncensored)
plt.figure(figsize=(8, 6))
plt.scatter(y_uncensored, train_predictions_knn, alpha=0.5)
plt.plot([y_uncensored.min(), y_uncensored.max()], [y_uncensored.min(), y_uncensored.max()], 'k--', lw=2)
plt.xlabel('Survival Time (y)')
plt.ylabel('Predicted Survival Time (y_hat)')
plt.title('y vs. y_hat Plot - k-NN Model')
plt.show()

submission_knn = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': final_predictions_knn})
submission_knn.to_csv('handle-missing-submission-knn-01.csv', index=False)

print("\nSubmissions saved.")
