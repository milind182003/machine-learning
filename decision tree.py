import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load datasets (Assuming features.csv, train.csv, stores.csv, and test.csv are available)
features_data = pd.read_csv('features.csv')
train_data = pd.read_csv('train.csv')
stores_data = pd.read_csv('stores.csv')
test_data = pd.read_csv('test.csv')

# Merge datasets
X = pd.merge(stores_data, pd.merge(train_data, features_data))
Xtest = pd.merge(stores_data, pd.merge(test_data, features_data))

# Drop Markdown columns due to missing values
X = X.drop(columns=["MarkDown1", 'MarkDown2', 'MarkDown3', "MarkDown4", 'MarkDown5'])
Xtest = Xtest.drop(columns=["MarkDown1", 'MarkDown2', 'MarkDown3', "MarkDown4", 'MarkDown5'])

# Fill missing values in test dataset
Xtest['Unemployment'].fillna(Xtest['Unemployment'].mean(), inplace=True)
Xtest['CPI'].fillna(Xtest['CPI'].mean(), inplace=True)

# Convert categorical variables to numerical
X['Type'] = X['Type'].map({'A': 2, 'B': 1, 'C': 0})
Xtest['Type'] = Xtest['Type'].map({'A': 2, 'B': 1, 'C': 0})

# Convert boolean attributes to integers
X['IsHoliday'] = X['IsHoliday'].astype(int)
Xtest['IsHoliday'] = Xtest['IsHoliday'].astype(int)

# Extract year and month from Date and encode
X['Year'] = pd.to_datetime(X['Date']).dt.year
X['Month'] = pd.to_datetime(X['Date']).dt.month
Xtest['Year'] = pd.to_datetime(Xtest['Date']).dt.year
Xtest['Month'] = pd.to_datetime(Xtest['Date']).dt.month

# Create month sin and month cos features
X['MonthSin'] = np.sin(2 * np.pi * X['Month'] / 12)
X['MonthCos'] = np.cos(2 * np.pi * X['Month'] / 12)
Xtest['MonthSin'] = np.sin(2 * np.pi * Xtest['Month'] / 12)
Xtest['MonthCos'] = np.cos(2 * np.pi * Xtest['Month'] / 12)

# Drop unnecessary columns
X = X.drop(["Date", "Year", "Month"], axis=1)
Xtest = Xtest.drop(["Date", "Year", "Month"], axis=1)

# Split data into training and validation sets
validation_ratio = 0.01
validation_indexes = np.random.choice(len(X), int(np.floor(len(X) * validation_ratio)), replace=False)
Xval = X.loc[validation_indexes]
X = X.drop(validation_indexes, axis=0)
y = X.pop('Weekly_Sales')
yval = Xval.pop('Weekly_Sales')

# Standardize data
def standardize_data(dataframe):
    numeric_columns = dataframe.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        if col != 'Store':
            dataframe[col] = (dataframe[col] - dataframe[col].mean()) / dataframe[col].std()

standardize_data(X)
standardize_data(Xval)
standardize_data(Xtest)

# Train linear regression model
linReg = LinearRegression()
linReg.fit(X, y)
linReg_predicted = linReg.predict(Xval)

# Train Lasso regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X.values, y.values)
lasso_predicted = lasso.predict(Xval.values)

# Train Random Forest model
randomForest = RandomForestRegressor(n_estimators=29)
randomForest.fit(X.values, y.values)
randomForest_predicted = randomForest.predict(Xval.values)

# Measure model accuracy
def rmse(predicted, target):
    return np.sqrt(np.mean((predicted - target) ** 2))

# Evaluate models
models = {'Linear Regression': linReg_predicted, 'Lasso Regression': lasso_predicted, 'Random Forest': randomForest_predicted}
metrics = ['RMSE', 'MAE', 'MSE', 'R-squared', 'Adjusted R-squared']
results = pd.DataFrame(columns=metrics)

n = len(Xval)  # Number of observations
p = X.shape[1]  # Number of predictors

for model_name, predictions in models.items():
    rmse_val = rmse(predictions, yval)
    mae_val = mean_absolute_error(yval, predictions)
    mse_val = mean_squared_error(yval, predictions)
    r_squared = r2_score(yval, predictions)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)  # Calculate adjusted R-squared
    results.loc[model_name] = [rmse_val, mae_val, mse_val, r_squared, adj_r_squared]

print(results)

# Data Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(15, 5))

# Scatter plot of actual vs predicted values for Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(yval, linReg_predicted, color='blue', label='Linear Regression')
plt.plot([min(yval), max(yval)], [min(yval), max(yval)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Linear Regression)')
plt.legend()

# Scatter plot of actual vs predicted values for Lasso Regression
plt.subplot(1, 3, 2)
plt.scatter(yval, lasso_predicted, color='green', label='Lasso Regression')
plt.plot([min(yval), max(yval)], [min(yval), max(yval)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Lasso Regression)')
plt.legend()

# Scatter plot of actual vs predicted values for Random Forest
plt.subplot(1, 3, 3)
plt.scatter(yval, randomForest_predicted, color='purple', label='Random Forest')
plt.plot([min(yval), max(yval)], [min(yval), max(yval)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Random Forest)')
plt.legend()

plt.tight_layout()
plt.show()
