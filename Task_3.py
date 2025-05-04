import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from math import sqrt

# Data Preparation Functions
def load_and_prepare_data():
    train_df = pd.read_csv("TrainData.csv", parse_dates=['TIMESTAMP'])[['TIMESTAMP', 'POWER']]
    solution_df = pd.read_csv("Solution.csv", parse_dates=['TIMESTAMP'])
    template_df = pd.read_csv("ForecastTemplate.csv", parse_dates=['TIMESTAMP'])

    combined_df = pd.concat([train_df, solution_df], ignore_index=True)
    combined_df.sort_values('TIMESTAMP', inplace=True)
    return train_df, solution_df, template_df, combined_df

def create_lag_features(df, window_size=24):
    df_lagged = df.copy()
    for lag in range(1, window_size + 1):
        df_lagged[f'lag_{lag}'] = df_lagged['POWER'].shift(lag)
    return df_lagged.dropna()

def scale_data(x, y):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    x_scaled = feature_scaler.fit_transform(x)
    y_scaled = target_scaler.fit_transform(y)
    return x_scaled, y_scaled, feature_scaler, target_scaler

# Model Training Functions
def train_models(x_train_scaled, y_train_scaled):
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(x_train_scaled, y_train_scaled)
    models['LR'] = lr

    # SVR
    svr = SVR(kernel='rbf')
    svr.fit(x_train_scaled, y_train_scaled.ravel())
    models['SVR'] = svr

    # ANN
    ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True)
    ann.fit(x_train_scaled, y_train_scaled.ravel())
    models['ANN'] = ann

    return models

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.RNN = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.RNN(x)
        return self.fc(out[:, -1, :])

def train_rnn_model(x_scaled, y_scaled, window_size=24, epochs=100):
    x_tensor = torch.from_numpy(x_scaled.reshape(-1, window_size, 1).astype(np.float32))
    y_tensor = torch.from_numpy(y_scaled.astype(np.float32))

    dataset = TensorDataset(x_tensor, y_tensor)
    train_len = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = RNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = sum(criterion(model(xb), yb).item() for xb, yb in val_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

    return model

# Forecasting and Evaluation
def forecast(models, x_pred_scaled, target_scaler, rnn_model=None, window_size=24):
    forecasts = {}
    for name, model in models.items():
        y_pred_scaled = model.predict(x_pred_scaled).reshape(-1, 1)
        forecasts[name] = target_scaler.inverse_transform(y_pred_scaled).flatten()

    if rnn_model:
        x_tensor = torch.from_numpy(x_pred_scaled.reshape(-1, window_size, 1).astype(np.float32))
        with torch.no_grad():
            y_pred_rnn_scaled = rnn_model(x_tensor).numpy()
        forecasts['RNN'] = target_scaler.inverse_transform(y_pred_rnn_scaled).flatten()

    return forecasts

def evaluate_forecasts(forecasts, true_values):
    results = []
    for name, preds in forecasts.items():
        rmse = sqrt(mean_squared_error(true_values, preds))
        smape = 100 * np.mean(2 * np.abs(preds - true_values) / (np.abs(preds) + np.abs(true_values) + 1e-3))
        results.append({'Model': name, 'RMSE': rmse, 'SMAPE': smape, 'Accuracy': 100 - smape})
    return pd.DataFrame(results).sort_values('RMSE')

def save_forecasts(forecasts, template_df):
    for name, preds in forecasts.items():
        output_df = pd.DataFrame({
            'TIMESTAMP': template_df['TIMESTAMP'],
            'POWER': preds
        })
        output_df.to_csv(f"ForecastTemplate3-{name}.csv", index=False)

def plot_forecasts(forecast_df, solution_df, preds1, preds2, name1, name2):
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df['TIMESTAMP'], preds1, label=f'{name1} Forecast')
    plt.plot(forecast_df['TIMESTAMP'], preds2, label=f'{name2} Forecast')
    plt.plot(solution_df['TIMESTAMP'], solution_df['POWER'], label='True Power', color='black')
    plt.xlabel("Time")
    plt.ylabel("POWER")
    plt.title(f"{name1} vs {name2} Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# Main Execution
def main():
    window_size = 24

    train_df, solution_df, template_df, combined_df = load_and_prepare_data()
    lagged_df = create_lag_features(combined_df, window_size)

    train_data = lagged_df[lagged_df['TIMESTAMP'] < '2013-11-01']
    forecast_data = lagged_df[
        (lagged_df['TIMESTAMP'] >= '2013-11-01') & (lagged_df['TIMESTAMP'] <= '2013-11-30 23:00:00')
    ]

    x_train = train_data.drop(columns=['TIMESTAMP', 'POWER'])
    y_train = train_data[['POWER']].values
    x_pred = forecast_data.drop(columns=['TIMESTAMP', 'POWER'])

    x_train_scaled, y_train_scaled, feature_scaler, target_scaler = scale_data(x_train, y_train)
    x_pred_scaled = feature_scaler.transform(x_pred)

    models = train_models(x_train_scaled, y_train_scaled)
    rnn_model = train_rnn_model(x_train_scaled, y_train_scaled, window_size)

    forecasts = forecast(models, x_pred_scaled, target_scaler, rnn_model, window_size)
    save_forecasts(forecasts, template_df)

    results = evaluate_forecasts(forecasts, solution_df['POWER'].values)
    print("Forecast RMSE and SMAPE Evaluation for Nov 2013")
    print(results)

    # Plotting the models
    plot_forecasts(forecast_data, solution_df, forecasts['LR'], forecasts['SVR'], 'LR', 'SVR')
    plot_forecasts(forecast_data, solution_df, forecasts['ANN'], forecasts['RNN'], 'ANN', 'RNN')

if __name__ == "__main__":
    main()