from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

knn_imputer = KNNImputer(n_neighbors=2)
features_to_impute = ['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']

train_data_clean.loc[:, features_to_impute] = knn_imputer.fit_transform(train_data_clean[features_to_impute])

X_uncensored = train_data_clean.drop(columns=['SurvivalTime', 'Censored'])
y_uncensored = train_data_clean['SurvivalTime']

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline.fit(X_uncensored, y_uncensored)

y_hat = pipeline.predict(X_uncensored)

mse = mean_squared_error(y_uncensored, y_hat)
print(f'Mean Squared Error (MSE Train): {mse:.3f}')

plt.figure(figsize=(8, 6))
plt.scatter(y_uncensored, y_hat, alpha=0.5)
plt.plot([y_uncensored.min(), y_uncensored.max()], [y_uncensored.min(), y_uncensored.max()], 'k--', lw=2)
plt.xlabel('Survival Time (y)')
plt.ylabel('Predicted Survival Time (y_hat)')
plt.title('y vs. y_hat Plot')
plt.show()

test_data.loc[:, features_to_impute] = knn_imputer.transform(test_data[features_to_impute])

final = pipeline.predict(test_data)

submission = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': final})
submission.to_csv('handle-missing-submission-knn-01.csv', index=False)

