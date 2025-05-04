import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as rnn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data
# Load datasets with datetime parsing
train_df = pd.read_csv("TrainData.csv", parse_dates=['TIMESTAMP'])
solution_df = pd.read_csv("Solution.csv", parse_dates=['TIMESTAMP'])

train_df = train_df[['TIMESTAMP', 'POWER']]
# solution_df = solution_df[['TIMESTAMP', 'POWER']]

# Create lag features
def create_lag_features(df, window_size=24):
    df_lagged = df.copy()
    for lag in range(1, window_size + 1):
        df_lagged[f'lag_{lag}'] = df_lagged['POWER'].shift(lag)
    return df_lagged.dropna()

# Create combined dataset for continuity in time series
combined_df = pd.concat([train_df, solution_df], ignore_index=True)
combined_df.sort_values('TIMESTAMP', inplace=True)

# Apply lag feature creation
window_size = 24
combined_lagged = create_lag_features(combined_df, window_size)

# Split the lagged data into train and forecast sets
train_sized = combined_lagged[combined_lagged['TIMESTAMP'] < pd.to_datetime('2013-11-01')]
forecast_df = combined_lagged[
    (combined_lagged['TIMESTAMP'] >= pd.to_datetime('2013-11-01')) &
    (combined_lagged['TIMESTAMP'] <= pd.to_datetime('2013-11-30 23:00:00'))
]

# Prepare features and target
x_train = train_sized.drop(columns=['TIMESTAMP', 'POWER'])
y_train = train_sized['POWER'].values.reshape(-1, 1)

# Scale features and target
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
x_train_scaled = feature_scaler.fit_transform(x_train)
y_train_scaled = target_scaler.fit_transform(y_train)

# Train LR model 
lr_model = LinearRegression()
lr_model.fit(x_train_scaled, y_train_scaled)

# Train Support Vector Regression using RBF Kernel
svr_model = SVR(kernel='rbf')
svr_model.fit(x_train_scaled, y_train_scaled.ravel())

# Train Artificial Neural Network Model
ann_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True)
ann_model.fit(x_train_scaled, y_train_scaled.ravel())

# Recurrent Neural Network Model Definition
class RNNModel(rnn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.RNN = rnn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = rnn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.RNN(x)
        out = self.fc(out[:, -1, :])
        return out

x_tensor = torch.from_numpy(x_train_scaled.reshape(-1, window_size, 1).astype(np.float32))
y_tensor = torch.from_numpy(y_train_scaled.astype(np.float32))

full_dataset = TensorDataset(x_tensor, y_tensor)
train_len = int(len(full_dataset) * 0.8)
val_len = len(full_dataset) - train_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Train RNN Model
rnn_model = RNNModel()
criterion = rnn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)

for epoch in range(100):
    rnn_model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        output = rnn_model(x_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # Validation
    rnn_model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            output = rnn_model(x_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()
            
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# Forecasting
x_pred = forecast_df.drop(columns=['TIMESTAMP', 'POWER'])
x_pred_scaled = feature_scaler.transform(x_pred)

y_pred_lr_scaled = lr_model.predict(x_pred_scaled)
y_pred_lr = target_scaler.inverse_transform(y_pred_lr_scaled).flatten()

y_pred_svr_scaled = svr_model.predict(x_pred_scaled)
y_pred_svr = target_scaler.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).flatten()

y_pred_ann_scaled = ann_model.predict(x_pred_scaled)
y_pred_ann = target_scaler.inverse_transform(y_pred_ann_scaled.reshape(-1, 1)).flatten()

x_pred_tensor = torch.from_numpy(x_pred_scaled.reshape(-1, window_size, 1).astype(np.float32))
rnn_model.eval()
with torch.no_grad():
    y_pred_rnn_scaled = rnn_model(x_pred_tensor).numpy() 
y_pred_rnn = target_scaler.inverse_transform(y_pred_rnn_scaled).flatten()


# Save forecasts
def save_forecast(filename, predictions):
    pd.DataFrame({
        'TIMESTAMP': solution_df['TIMESTAMP'],
        'POWER': predictions
    }).to_csv(filename, index=False)

save_forecast("ForecastTemplate3-LR.csv", y_pred_lr)
save_forecast("ForecastTemplate3-SVR.csv", y_pred_svr)
save_forecast("ForecastTemplate3-ANN.csv", y_pred_ann)
save_forecast("ForecastTemplate3-RNN.csv", y_pred_rnn)

# RMSE
true_power = solution_df['POWER'].values
rmse_lr = sqrt(mean_squared_error(true_power, y_pred_lr))
rmse_svr = sqrt(mean_squared_error(true_power, y_pred_svr))
rmse_ann = sqrt(mean_squared_error(true_power, y_pred_ann))
rmse_rnn = sqrt(mean_squared_error(true_power, y_pred_rnn))

rmse_results_df = pd.DataFrame({
    'Model': ['LR', 'SVR', 'ANN', 'RNN'],
    'RMSE': [rmse_lr, rmse_svr, rmse_ann, rmse_rnn]
})

print(f"Forecast RMSE Evaluation Nov 2013")
print(rmse_results_df.sort_values(by='RMSE'))

# Plot LR vs SVR
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['TIMESTAMP'], y_pred_lr, label=f'LR Forecast, RMSE={rmse_lr:.4f}', color='blue')
plt.plot(forecast_df['TIMESTAMP'], y_pred_svr, label=f'SVR Forecast, RMSE={rmse_svr:.4f}', color='red')
plt.plot(solution_df['TIMESTAMP'], solution_df['POWER'], label='True Power', color='black')
plt.xlabel("Time")
plt.ylabel("POWER")
plt.title("LR vs SVR Forecast for November 2013")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# Plot ANN vs RNN
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['TIMESTAMP'], y_pred_ann, label=f'ANN Forecast, RMSE={rmse_ann:.4f}', color='green')
plt.plot(forecast_df['TIMESTAMP'], y_pred_rnn, label=f'RNN Forecast, RMSE={rmse_rnn:.4f}', color='yellow')
plt.plot(solution_df['TIMESTAMP'], solution_df['POWER'], label='True Power', color='black')
plt.xlabel("Time")
plt.ylabel("POWER")
plt.title("ANN vs RNN Forecast for November 2013")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()