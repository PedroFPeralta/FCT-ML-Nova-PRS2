from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

# Remover linhas com valores nulos na coluna 'SurvivalTime' (alvo) e nas features
train_data_clean = train_data.dropna(subset=['SurvivalTime'])
print(train_data_clean.head())
# Remover qualquer linha com valores null nas features
train_data_clean = train_data_clean.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'])

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

final = pipeline.predict(test_data.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']))

submission = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': final})
submission.to_csv('baseline-submission-01.csv', index=False)
