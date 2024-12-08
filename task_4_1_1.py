import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

train_path = "train_data.csv"
test_path = "test_data.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Define the features to be imputed
features_to_impute = ['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']

# This time, unlabeled rows wont be removed
knn_imputer = KNNImputer(n_neighbors=2)  # Best option from task 3.1
knn_imputer.fit(train_data[features_to_impute])

train_data[features_to_impute] = knn_imputer.transform(train_data[features_to_impute])
test_data[features_to_impute] = knn_imputer.transform(test_data[features_to_impute])

train_data_clean = train_data.dropna(subset=['SurvivalTime'])

# Separate features (X) and target (y) for training
X_train = train_data_clean.drop(columns=['SurvivalTime', 'Censored'])
y_train = train_data_clean['SurvivalTime']

# Create a pipeline with feature scaling and linear regression
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  
    ('regressor', LinearRegression())  
])

# Train the model on the cleaned training data
pipeline.fit(X_train, y_train)

# Make predictions on the training data
y_hat = pipeline.predict(X_train)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_train, y_hat)
print(f'Mean Squared Error (MSE Train): {mse:.3f}')

# Plot predicted vs. actual survival times
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_hat, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel('Survival Time (y)')
plt.ylabel('Predicted Survival Time (y_hat)')
plt.title('y vs. y_hat Plot')
plt.show()

# Prepare the test data for predictions
X_test = test_data

# Make predictions on the test data
final_predictions = pipeline.predict(X_test)

# Save the predictions for submission
submission = pd.DataFrame({'id': test_data['id'], 'SurvivalTime': final_predictions})
submission.to_csv('imputed_predictions.csv', index=False)