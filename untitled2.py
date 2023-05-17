import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the 6-month dataset
data_subset = pd.read_csv('Covid19_Data_6month.csv')
data_subset['Date'] = pd.to_datetime(data_subset['Date'])
data_subset = data_subset.set_index('Date')

# Load the full-year dataset
data_full = pd.read_csv('Covid19_Data_12month.csv')
data_full['Date'] = pd.to_datetime(data_full['Date'])
data_full = data_full.set_index('Date')

# Check for missing dates
date_range = pd.date_range(start=data_subset.index.min(), end=data_subset.index.max())
missing_dates = date_range[~date_range.isin(data_subset.index)]
if len(missing_dates) > 0:
    print("Missing dates found: ", missing_dates)

# Fit the SARIMA model on the 6-month data
model = sm.tsa.statespace.SARIMAX(data_subset['Total_cases'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit()

# Generate predictions for the full year
pred_start_date = data_full.index.min()
pred_end_date = data_full.index.max()
y_pred = model_fit.predict(start=pred_start_date, end=pred_end_date)

# Create a new figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the 6-month prediction
ax.plot(data_subset.index, data_subset['Total_cases'], label='Actual Data (6 Months)')
ax.plot(data_full.index, y_pred, label='Predicted Data (Full Year)')

# Plot the full-year actual data
ax.plot(data_full.index, data_full['Total_cases'], label='Actual Data (Full Year)')

# Set labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Number of Cases')
ax.set_title('COVID-19 Trend Prediction for Santa Clara County')

# Show the legend
ax.legend()

# Display the plot
plt.grid(True)
plt.show()