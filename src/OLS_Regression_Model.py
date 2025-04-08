import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from statsmodels.nonparametric.smoothers_lowess import lowess

# Assuming data is already loaded and preprocessed with log transformations
# main_data contains the log-transformed variables

GPU_data = pd.read_csv("data/Cleaned_GPUs_KNN.csv")
main_data = GPU_data[["Memory_Bandwidth", "Memory_Speed", "Memory_Bus","Texture_Rate","Max_Power","L2_Cache"]].copy()

# 6. Log transform variables
main_data['Memory_Bandwidth'] = np.log1p(main_data['Memory_Bandwidth'])
main_data['Memory_Speed'] = np.log1p(main_data['Memory_Speed'])
main_data['L2_Cache'] = np.log1p(main_data['L2_Cache'])
main_data['Memory_Bus'] = np.log1p(main_data['Memory_Bus'])
main_data['Texture_Rate'] = np.log1p(main_data['Texture_Rate'])
main_data['Max_Power'] = np.log1p(main_data['Max_Power'])


# 1. Train-Test split
X = main_data.drop('Memory_Bandwidth', axis=1)  # Features
y = main_data['Memory_Bandwidth']               # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 2. Check for multicollinearity
X_with_const = sm.add_constant(X)  # Add constant term
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns[1:]  # Skip the constant term in variable names
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(1, X_with_const.shape[1])]  # Start from index 1 to skip constant
print("Variance Inflation Factors:")
print(vif_data)

# 3. Build statsmodels OLS model for detailed statistics
X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()
# Extract R-squared value
r_squared = model.rsquared
# Extract coefficient table as a DataFrame
summary_table = model.summary().tables[1]  # Table 1 contains coefficients
# Convert summary table to a DataFrame for logging
coefficients_df = pd.DataFrame(summary_table.data[1:], columns=summary_table.data[0])
coefficients_df.columns = ['Variable', 'Coef', 'Std Err', 't', 'P>|t|', '[0.025', '0.975]']
coefficients_df['P>|t|'] = coefficients_df['P>|t|'].astype(float).apply(lambda x: f"{x:.10f}")

# Log the required outputs
print("\nCoefficients Table:")
print(coefficients_df)
print("R-squared:", r_squared)
# 4. Check regression assumptions
# 4.1 Get predictions and residuals
y_train_pred = model.predict(X_train_sm)
residuals = y_train - y_train_pred

# Q-Q plot (similar to plot 2 in R)
from scipy import stats
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q')
plt.grid(True, alpha=0.3)
plt.savefig("Chart/Ket_Qua_Hoi_Quy/Normal QQ.png")
plt.close()

# 4.2 Create residuals plot
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.savefig("Chart/Ket_Qua_Hoi_Quy/Residuals vs Predicted.png")
plt.close()

#Scale location
plt.figure(figsize=(10, 6))
# calulate square of standardrized residuals
sqrt_std_resid = np.sqrt(np.abs(residuals))
# draw scatter plot
plt.scatter(y_train_pred, sqrt_std_resid, alpha=0.5)
# add LOWESS curve
# frac configure smooth level of the curve (0.5-0.7 are good values)
smoothed = lowess(sqrt_std_resid, y_train_pred, frac=0.6)
plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)
# Thêm tiêu đề và nhãn
plt.title('Scale-Location')
plt.xlabel('Fitted values')
plt.ylabel('√|Standardized residuals|')
plt.grid(True, alpha=0.3)
plt.savefig("Chart/Ket_Qua_Hoi_Quy/Scale Location.png")
plt.close()

# 5. Make predictions on test set
X_test_sm = sm.add_constant(X_test)
y_test_pred = model.predict(X_test_sm)

# 7. Calculate performance metrics
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print("\nModel Performance on logarit scale Test Data:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# 6. Transform predictions back from log scale
actual_value = np.expm1(y_test)
predicted_value = np.expm1(y_test_pred)

# Plot actual vs predicted on real scale data
plt.figure(figsize=(10, 6))
plt.scatter(actual_value, predicted_value)
plt.plot([min(actual_value), max(actual_value)], [min(actual_value), max(actual_value)], 'r--')
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True, alpha=0.3)
plt.savefig("Chart/Ket_Qua_Hoi_Quy/Actual vs Predicted.png")
plt.close()

# 7. Calculate performance metrics
mse = mean_squared_error(actual_value, predicted_value)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_value, predicted_value)
r2 = r2_score(actual_value, predicted_value)

print("\nModel Performance on real scale Test Data:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Calculate acceptable error ranges
error_5_percent = actual_value * 0.05
error_10_percent = actual_value*0.1

# Compare predictions within acceptable error range
within_5_percent = np.abs(predicted_value - actual_value) <= error_5_percent
within_10_percent = np.abs(predicted_value - actual_value) <= error_10_percent

accuracy_5 = np.mean(within_5_percent) * 100
accuracy_10 = np.mean(within_10_percent) * 100

print(f"Percentage of predictions within ±5% error: {accuracy_5:.2f}%")
print(f"Percentage of predictions within ±10% error: {accuracy_10:.2f}%")

