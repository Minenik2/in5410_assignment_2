import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as rnn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load and prepare data

# Load training data and solution
train_df = pd.read_csv("TrainData.csv")
solution_df = pd.read_csv("Solution.csv")

# Remove all columns from TrainData file except TIMESTAMP and Power
train_df = train_df[['TIMESTAMP', 'POWER']]

train_df['TIMESTAMP'] = pd.to_datetime(train_df['TIMESTAMP'], format='%Y%m%d %H:%M')
solution_df['TIMESTAMP'] = pd.to_datetime(solution_df['TIMESTAMP'], format='%Y%m%d %H:%M')

def create_window_size(df, size=24):
    data = df.copy()
    for i in range(1, size + 1):
        data[f'size_{i}'] = data['POWER'].shift(i)
    data.dropna(inplace=True)
    return data

size = 24
train_sized = create_window_size(train_df, size)

# Prepare train and test sets
x_train = train_sized.drop(columns=['TIMESTAMP', 'POWER'])
y_train = train_sized['POWER']

test_timestamps = solution_df['TIMESTAMP']
y_true = solution_df['POWER'].values

# Train Models

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# Support Vector Regression
svr_model = SVR(kernel='rbf')
svr_model.fit(x_train, y_train)

# Artificial Neural Network
ann_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)
ann_model.fit(x_train, y_train)

# Recurrent Neural Network
class RNNModel(rnn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = rnn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = rnn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)  # output: [batch, seq, hidden]
        out = self.fc(out[:, -1, :])  # use last time step's output
        return out

x_rnn = x_train.values.reshape(-1, 24, 1).astype(np.float32)
y_rnn = y_train.values.reshape(-1, 1).astype(np.float32)

train_dataset = TensorDataset(torch.from_numpy(x_rnn), torch.from_numpy(y_rnn))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

rnn_model = RNNModel()
criterion = rnn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    for x_batch, y_batch in train_loader:
        output = rnn_model(x_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
# Get last 24 power values before Nov 2013
initial_window = train_df[train_df['TIMESTAMP'] < '2013-11-01']['POWER'].values[-size:]

def rolling_forecast(model, initial_window, steps, reshape_input=False):
    history = initial_window.copy()
    predictions = []
    for _ in range(steps):
        input_vector = history[-size:]
        if reshape_input:
            input_vector = torch.tensor(input_vector.reshape(1, size, 1), dtype=torch.float32)
            with torch.no_grad():
                pred = model(input_vector).item()
        else:
            input_vector = input_vector.reshape(1, -1)
            pred = model.predict(input_vector)[0]
        predictions.append(pred)
        history = np.append(history[1:], pred)
    return predictions

steps = len(solution_df)
lr_preds = rolling_forecast(lr_model, initial_window, steps)
svr_preds = rolling_forecast(svr_model, initial_window, steps)
ann_preds = rolling_forecast(ann_model, initial_window, steps)
rnn_preds = rolling_forecast(rnn_model, initial_window, steps, reshape_input=True)

# Function to save forecast
def save_forecast(filename, timestamps, predictions):
    forecast_df = pd.DataFrame({
        'TIMESTAMP': timestamps,
        'POWER': predictions
    })
    forecast_df.to_csv(filename, index=False)
    
# Save forecasts
save_forecast("ForecastTemplate3-LR.csv", solution_df['TIMESTAMP'], lr_preds)
save_forecast("ForecastTemplate3-SVR.csv", solution_df['TIMESTAMP'], svr_preds)
save_forecast("ForecastTemplate3-ANN.csv", solution_df['TIMESTAMP'], ann_preds)
save_forecast("ForecastTemplate3-RNN.csv", solution_df['TIMESTAMP'], rnn_preds)

predictions = {
    'Linear Regression': lr_preds,
    'SVR': svr_preds,
    'ANN': ann_preds,
    'RNN': rnn_preds
}
# Plotting data
# Linear Regression and Support Vector Regression
# plt.figure(figsize=(15,5))
# plt.plot(solution_df['TIMESTAMP'], solution_df['POWER'], label='True Power', color='black')
# plt.plot(solution_df['TIMESTAMP'], lr_preds, label='Linear Regression', linestyle='--')
# plt.plot(solution_df['TIMESTAMP'], svr_preds, label='SVR', linestyle='--')
# plt.title("Wind Power Forecast - LR vs SVR")
# plt.xlabel("Timestamp")
# plt.ylabel("Power")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Artificial Neural Network and Recurrent Neural Network
# plt.figure(figsize=(15,5))
# plt.plot(solution_df['TIMESTAMP'], solution_df['POWER'], label='True Power', color='black')
# plt.plot(solution_df['TIMESTAMP'], ann_preds, label='ANN', linestyle='--')
# plt.plot(solution_df['TIMESTAMP'], rnn_preds, label='RNN', linestyle='--')
# plt.title("Wind Power Forecast - ANN vs RNN")
# plt.xlabel("Timestamp")
# plt.ylabel("Power")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the true values (always visible)
true_line, = ax.plot(test_timestamps, y_true, label='True Power', color='black', linewidth=2, alpha=0.8)

# Define model colors and plot predictions
model_colors = {
    'Linear Regression': 'blue',
    'SVR': 'orange',
    'ANN': 'green',
    'RNN': 'red'
}

lines = {}
for model_name, color in model_colors.items():
    y_pred = predictions[model_name]
    line, = ax.plot(test_timestamps, y_pred, label=f'{model_name} Prediction',
                    color=color)
    lines[model_name] = line

ax.set_title('Wind Power Forecasts vs True Power (Nov 2013)')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Normalized Wind Power')
ax.legend(loc='upper right')
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Create toggle functions
def make_toggle_func(model_name):
    def toggle(event):
        line = lines[model_name]
        line.set_visible(not line.get_visible())
        plt.draw()
    return toggle

# Add buttons to toggle each forecast curve
button_positions = [(0.1, 0.02), (0.3, 0.02), (0.5, 0.02), (0.7, 0.02)]
buttons = []
for (model_name, pos) in zip(lines.keys(), button_positions):
    ax_button = plt.axes([pos[0], pos[1], 0.1, 0.04])
    button = Button(ax_button, model_name)
    button.on_clicked(make_toggle_func(model_name))
    buttons.append(button)

plt.show()
    
# Compare forecasting accuracy RMSE
true_vals = solution_df['POWER'].values

rmse_lr = sqrt(mean_squared_error(true_vals, lr_preds))
rmse_svr = sqrt(mean_squared_error(true_vals, svr_preds))
rmse_ann = sqrt(mean_squared_error(true_vals, ann_preds))
rmse_rnn = sqrt(mean_squared_error(true_vals, rnn_preds))

# Display results in a table
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'SVR', 'ANN', 'RNN'],
    'RMSE': [rmse_lr, rmse_svr, rmse_ann, rmse_rnn]
})

print("\nForecasting Accuracy (RMSE):")
print(results_df.sort_values(by="RMSE"))