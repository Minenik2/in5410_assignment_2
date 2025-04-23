import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib.widgets import Button

# Load training data
train = pd.read_csv('TrainData.csv', parse_dates=['TIMESTAMP'])
X_train = train[['WS10']].values
y_train = train['POWER'].values

# Load weather forecast input (for Nov 2013)
weather_forecast = pd.read_csv('WeatherForecastInput.csv', parse_dates=['TIMESTAMP'])
X_test = weather_forecast[['WS10']].values
timestamps_test = weather_forecast['TIMESTAMP']

# Load true values for Nov 2013
solution = pd.read_csv('Solution.csv', parse_dates=['TIMESTAMP'])
y_true = solution['POWER'].values

# Load ForecastTemplate.csv
template = pd.read_csv('ForecastTemplate.csv', parse_dates=['TIMESTAMP'])

# Dictionary to store models and filenames
models = {
    'LR': (LinearRegression(), 'ForecastTemplate1-LR.csv'),
    'kNN': (KNeighborsRegressor(n_neighbors=5), 'ForecastTemplate1-kNN.csv'),
    'SVR': (SVR(kernel='rbf', C=100, epsilon=0.1), 'ForecastTemplate1-SVR.csv'),
    'NN': (MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42), 'ForecastTemplate1-NN.csv')
}
# Prepare to store predictions
predictions = {}


# Train, predict, and evaluate
for name, (model, filename) in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save prediction to template
    forecast_df = template.copy()
    forecast_df['FORECAST'] = y_pred
    forecast_df.to_csv(filename, index=False)
    
    # Store predictions for plotting
    predictions[name] = y_pred

    # Evaluate RMSE
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} RMSE: {rmse:.4f}")
    
    # Evaluating SMAPE (more stable MAPE when values are close to zero)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-3))
    accuracy = 100 - smape
    print(f"{name} Prediction Accuracy: {accuracy:.2f}%")

#
# PLOTTING DATA
#

# Plotting the prediction accuracy
fig, ax = plt.subplots(figsize=(10, 6))

# Plot actual values
ax.plot(timestamps_test, y_true, label='True Power', color='black', linewidth=2, alpha=0.8)

# Plot predicted values for each model
lines = {}
for name, y_pred in predictions.items():
    line, = ax.plot(timestamps_test, y_pred, label=f'{name} Prediction')
    lines[name] = line

# Formatting the plot
plt.title('Wind Power Forecasts vs Actuals for November 2013')
plt.xlabel('Timestamp')
plt.ylabel('Normalized Wind Power')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Function to toggle visibility of a model's plot
def toggle_visibility(label):
    line = lines[label]
    line.set_visible(not line.get_visible())
    plt.draw()

# Create buttons to toggle visibility
button_positions = [(0.1, 0.02), (0.3, 0.02), (0.5, 0.02), (0.7, 0.02)]  # positions for buttons
buttons = []

for i, (name, line) in enumerate(lines.items()):
    ax_button = plt.axes([button_positions[i][0], button_positions[i][1], 0.1, 0.04])  # x, y, width, height
    button = Button(ax_button, name)
    button.on_clicked(lambda event, label=name: toggle_visibility(label))
    buttons.append(button)

# Show the plot
plt.show()