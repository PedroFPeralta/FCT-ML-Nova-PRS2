from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
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
censored = train_data_clean['Censored']

def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

def custom_scorer(estimator, X, y):
    y_hat = estimator.predict(X)
    return error_metric(y, y_hat, censored.loc[X.index])

pipeline.fit(X_uncensored, y_uncensored)

y_hat = pipeline.predict(X_uncensored)

final_error = error_metric(y_uncensored, y_hat, censored)
print(f'Error no conjunto de treino: {final_error:.3f}')

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
