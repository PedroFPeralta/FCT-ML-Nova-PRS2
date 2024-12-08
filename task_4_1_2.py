import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.manifold import Isomap
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

class FrozenTransformer(BaseEstimator):
    def __init__(self, fitted_transformer):
        self.fitted_transformer = fitted_transformer

    def __getattr__(self, name):
        return getattr(self.fitted_transformer, name)

    def __sklearn_clone__(self):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.fitted_transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fitted_transformer.transform(X)

train_path = "train_data.csv"
test_path = "test_data.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

features_to_impute = ['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']

knn_imputer = KNNImputer(n_neighbors=2)
knn_imputer.fit(train_data[features_to_impute])

train_data[features_to_impute] = knn_imputer.transform(train_data[features_to_impute])
test_data[features_to_impute] = knn_imputer.transform(test_data[features_to_impute])

X_all = train_data.drop(columns=['SurvivalTime', 'Censored'], errors='ignore')

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# train isomap
isomap = Isomap(n_components=5)
isomap.fit(X_all_scaled)

frozen_isomap = FrozenTransformer(isomap)

# Remove rows with missing target ('SurvivalTime') from the training data
train_data_clean = train_data.dropna(subset=['SurvivalTime'])

X_train = train_data_clean.drop(columns=['SurvivalTime', 'Censored'])
y_train = train_data_clean['SurvivalTime']

# create pipeline
pipeline = make_pipeline(
    StandardScaler(),
    frozen_isomap,
    LinearRegression()
)

pipeline.fit(X_train, y_train)

y_hat_iso = pipeline.predict(X_train)
mse_iso = mean_squared_error(y_train, y_hat_iso)
print(f'Mean Squared Error (MSE with Isomap): {mse_iso:.3f}')

plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_hat_iso, alpha=0.5, label='With Isomap')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel('Survival Time (y)')
plt.ylabel('Predicted Survival Time (y_hat)')
plt.title('y vs. y_hat Plot with Isomap')
plt.legend()
plt.show()

X_test = test_data
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

final_predictions_iso = pipeline.predict(X_test_scaled)

submission_iso = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': final_predictions_iso})
submission_iso.to_csv('isomap_predictions.csv', index=False)