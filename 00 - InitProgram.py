import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

train_path = "train_data.csv"
test_path = "test_data.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Função de erro censurado (cMSE)
def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]

# Verificar se há NaN em SurvivalTime
print(f"Valores ausentes em y (SurvivalTime): {train_data['SurvivalTime'].isna().sum()}")

# Imputar valores ausentes de 'SurvivalTime' com a média
imputer_y = SimpleImputer(strategy='mean')
train_data['SurvivalTime'] = imputer_y.fit_transform(train_data[['SurvivalTime']])

X = train_data.drop(columns=['SurvivalTime'])
y = train_data['SurvivalTime']
c = train_data['Censored']

genetic_risk = ['GeneticRisk']
comorbidity_index = ['ComorbidityIndex']
treatment_response = ['TreatmentResponse']

preprocessor = ColumnTransformer(
    transformers=[
        ('genetic_risk', SimpleImputer(strategy='mean'), genetic_risk),
        ('comorbidity_index', SimpleImputer(strategy='most_frequent'), comorbidity_index),
        ('treatment_response', SimpleImputer(strategy='constant', fill_value=0), treatment_response)
    ])

def cMSE_scorer(estimator, X, y):
    y_hat = estimator.predict(X)
    return error_metric(y, y_hat, c)

custom_scorer = make_scorer(cMSE_scorer, greater_is_better=False)

model = LinearRegression()
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

scores = cross_val_score(pipeline, X, y, cv=5, scoring=custom_scorer)

print(f"Pontuação média da validação cruzada: {-scores.mean():.3f}")
