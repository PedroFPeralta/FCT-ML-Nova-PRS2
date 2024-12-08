import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_path = "train_data.csv"
test_path = "test_data.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data = train_data.dropna(subset=['SurvivalTime', 'Censored'])

train_data['y_lower'] = train_data['SurvivalTime']
train_data['y_upper'] = np.where(train_data['Censored'] == 1, train_data['SurvivalTime'], -1)

train_data = train_data.dropna(subset=['y_lower', 'y_upper'])

train, valid = train_test_split(train_data, test_size=0.2, random_state=42)

features = train.columns.difference(['SurvivalTime', 'Censored', 'y_lower', 'y_upper'], sort=False)

categorical_features = []

train_pool = Pool(train[features], label=train[['y_lower', 'y_upper']], cat_features=categorical_features)
valid_pool = Pool(valid[features], label=valid[['y_lower', 'y_upper']], cat_features=categorical_features)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.01,
    depth=8,
    loss_function='SurvivalAft:dist=Normal',
    eval_metric='SurvivalAft',
    verbose=50
)

model.fit(train_pool, eval_set=valid_pool)

valid_predictions = model.predict(valid_pool, prediction_type='Exponent')

valid_true = valid['y_lower']
mse = mean_squared_error(valid_true, valid_predictions)
print(f"\nMean Squared Error on Validation Set: {mse:.4f}")

test_pool = Pool(test_data[features], cat_features=categorical_features)

test_predictions = model.predict(test_pool, prediction_type='Exponent')

submission = pd.DataFrame({'id': test_data['id'], 'PredictedSurvivalTime': test_predictions})
submission.to_csv('CatBoost_SurvivalAFT_submission.csv', index=False)
print("\nPredictions saved to 'CatBoost_SurvivalAFT_submission.csv'.")

plt.figure(figsize=(8, 6))
plt.scatter(valid_true, valid_predictions, alpha=0.5, label="Predicted vs Actual")
plt.plot([valid_true.min(), valid_true.max()], [valid_true.min(), valid_true.max()], 'k--', lw=2, label="Perfect Prediction")
plt.xlabel('Actual Survival Time (y)')
plt.ylabel('Predicted Survival Time (y_hat)')
plt.title('y vs. y_hat Plot - CatBoostRegressor (SurvivalAFT)')
plt.legend()
plt.show()


