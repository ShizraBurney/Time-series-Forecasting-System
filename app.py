from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from joblib import load as joblib_load
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
import prophet
from pandas import date_range, to_datetime
from sklearn.preprocessing import MinMaxScaler
from traceback import format_exc
from tensorflow.keras.initializers import Orthogonal
from database import init_db, insert_forecast, get_forecasts
from tensorflow.keras.layers import LSTM


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import sqlite3

app = Flask(__name__)

model_files = {
    'finance_arima': 'finance_arima.pkl',
    'finance_ann': 'finance_ann.h5',
    'finance_ann_scaler': 'finance_ann_scaler.pkl',
    'finance_hybrid_arima_ann': 'finance_hybrid_arima_ann_ann.h5',
    'finance_hybrid_arima_ann_scaler': 'finance_hybrid_arima_ann_scaler.pkl',
    'finance_hybrid_arima_ann_arima': 'finance_hybrid_arima_ann_arima.pkl',
    'finance_sarima': 'finance_sarima.pkl',
    'finance_ets': 'finance_ets.pkl',
    'finance_prophet': 'finance_prophet.pkl',
    'finance_svr': 'finance_svr_model.pkl',
    'finance_svr_scaler': 'finance_svr_scaler.pkl',
    'finance_lstm': 'finance_lstm.h5',
    'finance_lstm_scaler': 'finance_lstm_scaler.pkl',
}

models = {}
scalers = {}

class CustomLSTM(LSTM):
    def __init__(self, units, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(units, **kwargs)

custom_objects = {'Orthogonal': Orthogonal, 'LSTM': CustomLSTM}

for key, file in model_files.items():
    if file.endswith('.h5'):
        models[key] = load_model(file, custom_objects=custom_objects)  # Load Keras models with custom LSTM layer
    elif file.endswith('.pkl'):
        models[key] = joblib_load(file)  # Load ARIMA and other models
        if 'scaler' in key:
            scalers[key] = joblib_load(file)  # Load scalers

def generate_forecasts(data):
    dataset = data['dataset']
    model_type = data['model']
    model_key = f"{dataset}_{model_type}"  # e.g., 'finance_hybrid_arima_ann'
    model = models.get(model_key)
    scaler = scalers.get(f"{dataset}_{model_type}_scaler")  # Load corresponding scaler

    if not model:
        raise ValueError(f"No model found for key {model_key}")

    # Load the dataset
    df = pd.read_csv(f"{dataset}.csv")  # Adjust path as necessary
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    last_values = df[['Close']].values[-1].reshape(1, -1)  # Assuming 'Close' column needs to be used

    if 'arima' in model_key:
        # ARIMA model
        steps = 5
        forecast_result = model.get_forecast(steps=steps)
        dates = date_range(start=last_date, periods=steps, freq='M')
        forecasted_values = forecast_result.predicted_mean
        results = [{"date": str(date.strftime('%Y-%m')), "forecasted_values": float(value)} for date, value in zip(dates, forecasted_values)]
    
    elif 'sarima' in model_key:
        steps = 5
        forecast_result = model.get_forecast(steps=steps)
        dates = date_range(start=last_date, periods=steps, freq='M')
        forecasted_values = forecast_result.predicted_mean
        results = [{"date": str(date.strftime('%Y-%m')), "forecasted_values": float(value)} for date, value in zip(dates, forecasted_values)]

    elif 'ann' in model_key:
        scaler_key = f"{dataset}_{model_type}_scaler"
        scaler = scalers.get(scaler_key)
        if not scaler:
            raise ValueError(f"Scaler not found for key {scaler_key}")

        last_values_scaled = scaler.transform(last_values)
        predictions = []
        for _ in range(5):  # Forecast 5 steps
            prediction = model.predict(last_values_scaled)
            predictions.append(prediction[0][0])
            last_values_scaled = prediction  # Update last_value with the predicted one for the next step
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        dates = date_range(start=last_date, periods=len(predictions), freq='M')
        results = [{"date": str(date.strftime('%Y-%m')), "forecasted_values": float(value)} for date, value in zip(dates, predictions)]
    
    elif 'ets' in model_key:
        steps = 5
        try:
            forecasted_values = model.forecast(steps=steps)  # Use the correct forecasting method
            dates = date_range(start=last_date, periods=steps, freq='M')
            results = [{"date": str(date.strftime('%Y-%m')), "forecasted_values": float(value)} for date, value in zip(dates, forecasted_values)]
        except Exception as e:
            print("Failed to forecast with ETS model:", str(e))
            raise
    
    elif 'prophet' in model_key:
        # Prophet model
        future = model.make_future_dataframe(periods=5, freq='M')
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat']].tail(5)  # Get the last 5 predictions
        results = [{"date": str(date.strftime('%Y-%m')), "forecasted_values": float(value)} for date, value in zip(forecast['ds'], forecast['yhat'])]
    
    elif 'svr' in model_key:
        scaler_key = f"{dataset}_{model_type}_scaler"
        scaler = scalers.get(scaler_key)
        if not scaler:
            raise ValueError(f"Scaler not found for key {scaler_key}")

        last_values_scaled = scaler.transform(last_values)
        predictions = []
        for _ in range(5):  # Forecast 5 steps
            prediction = model.predict(last_values_scaled)
            predictions.append(prediction[0])
            last_values_scaled = prediction.reshape(1, -1)  # Update last_value with the predicted one for the next step
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        dates = date_range(start=last_date, periods=len(predictions), freq='M')
        results = [{"date": str(date.strftime('%Y-%m')), "forecasted_values": float(value)} for date, value in zip(dates, predictions)]

    elif 'lstm' in model_key:
        scaler_key = f"{dataset}_{model_type}_scaler"
        scaler = scalers.get(scaler_key)
        if not scaler:
            raise ValueError(f"Scaler not found for key {scaler_key}")

        last_values_scaled = scaler.transform(last_values)
        last_values_scaled = last_values_scaled.reshape((last_values_scaled.shape[0], 1, last_values_scaled.shape[1]))  # Reshape for LSTM input
        predictions = []
        for _ in range(5):  # Forecast 5 steps
            prediction = model.predict(last_values_scaled)
            predictions.append(prediction[0][0])
            last_values_scaled = np.roll(last_values_scaled, -1, axis=1)  # Shift data
            last_values_scaled[0, -1, 0] = prediction  # Insert new prediction
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        dates = date_range(start=last_date, periods=len(predictions), freq='M')
        results = [{"date": str(date.strftime('%Y-%m')), "forecasted_values": float(value)} for date, value in zip(dates, predictions)]

    # Save forecasts to the database
    for result in results:
        insert_forecast(dataset, model_type, result['date'], result['forecasted_values'])

    return {"forecast": results}

# Dash app integration
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/')

dash_app.layout = html.Div([
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Finance', 'value': 'finance'},
            {'label': 'Energy', 'value': 'energy'},
            {'label': 'Environment', 'value': 'environment'}
        ],
        value='finance'
    ),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'ARIMA', 'value': 'arima'},
            {'label': 'SARIMA', 'value': 'sarima'},
            {'label': 'ETS', 'value': 'ets'},
            {'label': 'Prophet', 'value': 'prophet'},
            {'label': 'SVR', 'value': 'svr'},
            {'label': 'LSTM', 'value': 'lstm'},
            {'label': 'ANN', 'value': 'ann'}
        ],
        value='arima'
    ),
    html.H2(id='time-series-title'),
    dcc.Graph(id='time-series-graph'),
    html.H2(id='forecast-title'),
    dcc.Graph(id='forecast-graph'),
    html.H2(id='residuals-title'),
    dcc.Graph(id='residuals-graph'),
    html.H2(id='accuracy-title'),
    dcc.Graph(id='accuracy-graph')
])


@dash_app.callback(
    [Output('time-series-title', 'children'),
     Output('time-series-graph', 'figure'),
     Output('forecast-title', 'children'),
     Output('forecast-graph', 'figure'),
     Output('residuals-title', 'children'),
     Output('residuals-graph', 'figure'),
     Output('accuracy-title', 'children'),
     Output('accuracy-graph', 'figure')],
    [Input('dataset-dropdown', 'value'),
     Input('model-dropdown', 'value')]
)
def update_graphs(dataset, model):
    # Fetch historical data
    df = pd.read_csv(f"{dataset}.csv")  # Adjust path as necessary

    # Fetch forecasted data
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''
    SELECT date, forecasted_value FROM forecasts WHERE dataset = ? AND model = ?
    ''', (dataset, model))
    forecast_data = c.fetchall()
    conn.close()

    dates = [row[0] for row in forecast_data]
    forecasted_values = [row[1] for row in forecast_data]

    # Historical time series data
    time_series_fig = go.Figure()
    time_series_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Data'))
    time_series_title = f"Time Series Data for {dataset.capitalize()} Dataset"

    # Forecasted data
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=dates, y=forecasted_values, mode='lines', name='Forecast'))
    forecast_title = f"Forecast using {model.upper()} Model"

    # Residuals (Actual - Forecast)
    residuals = df['Close'].values[-len(forecasted_values):] - forecasted_values
    residuals_fig = go.Figure()
    residuals_fig.add_trace(go.Scatter(x=dates, y=residuals, mode='lines', name='Residuals'))
    residuals_title = f"Residuals for {model.upper()} Model on {dataset.capitalize()} Dataset"

    # Accuracy metrics
    # Placeholder for actual accuracy metrics implementation
    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(x=dates, y=residuals**2, mode='lines', name='Squared Residuals'))  # Example metric
    accuracy_title = f"Accuracy Metrics for {model.upper()} Model on {dataset.capitalize()} Dataset"

    return time_series_title, time_series_fig, forecast_title, forecast_fig, residuals_title, residuals_fig, accuracy_title, accuracy_fig




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/generate_forecast', methods=['POST'])
def generate_forecast_route():
    try:
        data = request.get_json()
        forecasts = generate_forecasts(data)
        return jsonify(forecasts)
    except Exception as e:
        error_message = format_exc()
        app.logger.error(f"Error processing request: {error_message}")
        return jsonify({'error': str(e), 'trace': error_message}), 500

@app.route('/get_forecasts', methods=['GET'])
def get_forecasts_route():
    try:
        dataset = request.args.get('dataset')
        model = request.args.get('model')
        forecasts = get_forecasts(dataset, model)
        return jsonify([{"date": date, "forecasted_value": value} for date, value in forecasts])
    except Exception as e:
        error_message = format_exc()
        app.logger.error(f"Error processing request: {error_message}")
        return jsonify({'error': str(e), 'trace': error_message}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
