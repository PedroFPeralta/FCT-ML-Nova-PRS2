from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd

train_path = "train_data.csv"
test_path = "test_data.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data_clean = train_data.dropna(subset=['SurvivalTime'])

X = train_data_clean.drop(columns=['SurvivalTime', 'Censored'])
y = train_data_clean['SurvivalTime']

test_data_clean = test_data

print("\nHistGradientBoostingRegressor Evaluation:")

hist_model = HistGradientBoostingRegressor()

hist_model.fit(X, y)

cv_scores_hist = cross_val_score(hist_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Mean CV MSE for HistGradientBoostingRegressor: {-cv_scores_hist.mean():.3f}")

hist_predictions = hist_model.predict(test_data_clean)

submission_hist = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': hist_predictions})
submission_hist.to_csv('HistGBR_submission.csv', index=False)
print("\nPredictions saved for HistGradientBoostingRegressor.")

train_predictions_hist = hist_model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(y, train_predictions_hist, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Survival Time (y)')
plt.ylabel('Predicted Survival Time (y_hat)')
plt.title('y vs. y_hat Plot - HistGradientBoostingRegressor')
plt.show()