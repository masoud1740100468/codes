
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from gplearn.functions import make_function

# Define a safe square root function that handles non-negative inputs
def sqrt(x):
    return np.sqrt(np.abs(x))  # Absolute value ensures non-negative input

# Define a cubic root function
def cbrt(x):
    return np.cbrt(x)

# Define a power function (e.g., square, i.e., x^2)
def power(x):
    return np.power(x, 2)

# Define a safe multiplication function that always uses positive coefficients
def safe_mul(x, y):
    return np.abs(x) * np.abs(y)  # Absolute value ensures positive coefficients

# Wrap custom functions for gplearn
sqrt_function = make_function(function=sqrt, name="sqrt", arity=1)
cbrt_function = make_function(function=cbrt, name="cbrt", arity=1)
power_function = make_function(function=power, name="power", arity=1)
safe_mul_function = make_function(function=safe_mul, name="safe_mul", arity=2)

# Step 1: Load the data
data = pd.read_csv('C:/XAI/codes/Rpacks/DALEX - enhanced/new-gp-relation/data-for-gp.csv')

# Step 2: Prepare the data
X = data.iloc[:, 1:].values  # All columns except the first one (inputs)
y = data.iloc[:, 0].values    # First column (target)

# Step 3: Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.4,shuffle=False)

# Step 5: Configure Symbolic Regressor
gp = SymbolicRegressor(
    population_size=150,
    generations=200,
    stopping_criteria=0.00001,
    p_crossover=0.3,
    p_subtree_mutation=0.2,
    p_hoist_mutation=0.1,
    p_point_mutation=0.2,
    init_depth=(5, 10),
    metric='mse',
    parsimony_coefficient=0.0001,  # Lowered to encourage using all features
    random_state=0,
    tournament_size=30,
    function_set=["div", sqrt_function, cbrt_function, power_function, safe_mul_function]
)

# Step 6: Fit the model
gp.fit(X_train, y_train)

# Step 7: Evaluate the model on test data
y_pred_normalized = gp.predict(X_test)

# Step 8: Inverse transform predictions to original scale
y_pred = scaler_y.inverse_transform(y_pred_normalized.reshape(-1, 1)).ravel()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Step 9: Output results
print("Best program:\n", gp._program)
print("Test Predictions (original scale):\n", y_pred)
print("Test Targets (original scale):\n", y_test_original)
print("Test MAE (original scale):", np.mean(np.abs(y_pred - y_test_original)))

# Step 10: Calculate and print R² score for test data
r2_test = r2_score(y_test_original, y_pred)
print("R² Score (test set):", r2_test)

# Step 11: Calculate and print R² score for training data
y_train_pred_normalized = gp.predict(X_train)
y_train_pred = scaler_y.inverse_transform(y_train_pred_normalized.reshape(-1, 1)).ravel()
y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
r2_train = r2_score(y_train_original, y_train_pred)
print("R² Score (train set):", r2_train)



# Output the best program
print("Best program:\n", gp._program)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Calculate performance metrics for the test set
r2_test = r2_score(y_test_original, y_pred)
mae_test = mean_absolute_error(y_test_original, y_pred)
mse_test = mean_squared_error(y_test_original, y_pred)
rmse_test = np.sqrt(mse_test)

# Calculate performance metrics for the train set
r2_train = r2_score(y_train_original, y_train_pred)
mae_train = mean_absolute_error(y_train_original, y_train_pred)
mse_train = mean_squared_error(y_train_original, y_train_pred)
rmse_train = np.sqrt(mse_train)

# Output results for the test set
print("Test Set Performance Metrics:")
print(f"R² Score (test set): {r2_test:.4f}")
print(f"Mean Absolute Error (MAE, test set): {mae_test:.4f}")
print(f"Mean Squared Error (MSE, test set): {mse_test:.4f}")
print(f"Root Mean Squared Error (RMSE, test set): {rmse_test:.4f}")

# Output results for the train set
print("\nTrain Set Performance Metrics:")
print(f"R² Score (train set): {r2_train:.4f}")
print(f"Mean Absolute Error (MAE, train set): {mae_train:.4f}")
print(f"Mean Squared Error (MSE, train set): {mse_train:.4f}")
print(f"Root Mean Squared Error (RMSE, train set): {rmse_train:.4f}")
# Save predictions and observations to an Excel file
train_data = pd.DataFrame({
    'Observed': y_train_original,
    'Predicted': y_train_pred
})

test_data = pd.DataFrame({
    'Observed': y_pred,
    'Predicted': y_test_original
})

with pd.ExcelWriter('predictions_observations-shear-stress-new-gp-equation-safety.xlsx') as writer:
    train_data.to_excel(writer, sheet_name='Train Data', index=False)
    test_data.to_excel(writer, sheet_name='Test Data', index=False)

print("\nPredictions and observations have been saved to 'predictions_observations-shear-stress-new-gp-equation-safety.xlsx'")