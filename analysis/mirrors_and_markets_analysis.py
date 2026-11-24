"""
Mirrors and Markets - analysis script (NEOMA M.Sc dissertation, 2024)
Yann Simons

Last saved version of the analysis notebook (1 September 2024), exported
from Google Colab. Code below is unchanged except for the seaborn import,
which was missing from the saved cell.

Pipeline: GME close/volume and VIX (Yahoo Finance, 2020-01-01 to 2021-12-31)
+ daily Reddit sentiment (mean VADER compound score of post titles) ->
descriptive statistics, correlation matrix, OLS (Close ~ Volume + Sentiment
+ VIX), wavelet transform of Close vs Sentiment.

Notes for this repository:
- Sentiment inputs come from the public-domain "GameStop Reddit Dataset"
  (Han & Iliadis, Temple University). Only r/GME, r/DDintoGME and
  r/GMEJungle carry a `compound` score; the r/Superstonk file has none.
  The daily average of those scores is provided in
  data/daily_reddit_sentiment.csv (column `sentiment_combined`); raw posts
  are not redistributed here.
- The run reported in the dissertation (2 September 2024) also added three
  price lags (Lag1-Lag3), an LSTM forecast and a wavelet-coherence heatmap;
  that final notebook state was not preserved.
"""
import seaborn as sns  # missing in the saved notebook

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pywt import cwt, morlet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to fetch stock data and VIX data
def fetch_data(ticker, vix_ticker='^VIX', start_date='2020-01-01', end_date='2021-12-31'):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    vix_data = yf.download(vix_ticker, start=start_date, end=end_date)['Close']
    return stock_data[['Close', 'Volume']], vix_data

# Load and combine sentiment data from multiple CSV files
def load_combine_sentiment_data(filepaths):
    frames = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        frames.append(df['compound'])
    combined_sentiment = pd.concat(frames, axis=0).groupby(level=0).mean()
    return combined_sentiment

# Descriptive statistics and visualization
def descriptive_statistics(data):
    desc = data.describe()
    plt.figure(figsize=(12, 7))
    plt.subplot(211)
    plt.plot(data['Close'])
    plt.title('Stock Price Over Time')
    plt.subplot(212)
    plt.bar(data.index, data['Volume'])
    plt.title('Trading Volume Over Time')
    plt.tight_layout()
    plt.show()
    return desc

# Correlation analysis and visualization
def correlation_analysis(data):
    correlation = data.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    return correlation

# Regression analysis
def regression_analysis(data):
    Y = data['Close']
    X = sm.add_constant(data[['Volume', 'Sentiment', 'VIX']])
    model = sm.OLS(Y, X).fit()
    return model

# Wavelet coherence visualization
def wavelet_coherence_analysis(series1, series2):
    scales = np.arange(1, 128)
    cwtmatr1, _ = cwt(series1.values, scales, 'cmor')
    cwtmatr2, _ = cwt(series2.values, scales, 'cmor')
    coherence = np.abs(np.corrcoef(cwtmatr1, cwtmatr2)[0, 1])
    plt.figure(figsize=(10, 4))
    plt.plot(coherence)
    plt.title('Wavelet Coherence between Sentiment and Stock Prices')
    plt.show()
    return coherence

# Main execution block
if __name__ == "__main__":
    stock_data, vix_data = fetch_data('GME')
    sentiment_files = ['r_ddintogme.csv', 'r_gme.csv', 'r_gmejungle.csv', 'r_superstonk.csv']
    sentiment_data = load_combine_sentiment_data(sentiment_files)

    data = pd.DataFrame({
        'Close': stock_data['Close'],
        'Volume': stock_data['Volume'],
        'VIX': vix_data,
        'Sentiment': sentiment_data.reindex(stock_data.index).fillna(method='bfill')
    }).dropna()

    descriptive_stats = descriptive_statistics(data)
    corr_matrix = correlation_analysis(data)
    regression_results = regression_analysis(data)
    print(regression_results.summary())

    wavelet_result = wavelet_coherence_analysis(data['Close'], data['Sentiment'])
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pywt import cwt, ricker
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to fetch stock data and VIX data
def fetch_data(ticker, start_date='2020-01-01', end_date='2021-01-01'):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data[['Close', 'Volume']]

# Enhanced Feature Engineering for LSTM
def create_features(data):
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)
    data['Lag3'] = data['Close'].shift(3)
    data.dropna(inplace=True)
    return data

# Predictive Modeling using LSTM
def predictive_modeling(features, target):
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return scaler, model, X_test, y_test, predictions, mse

# Visualization of Predictive Modeling Results
def visualize_predictions(scaler, X_test, y_test, predictions):
    actual_prices = scaler.inverse_transform(y_test)
    predicted_prices = scaler.inverse_transform(predictions)

    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, label='Actual Prices')
    plt.plot(predicted_prices, label='Predicted Prices', alpha=0.7)
    plt.title('Stock Price Prediction with LSTM')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Wavelet Coherence Heatmap Visualization
def plot_wavelet_coherence_heatmap(coherence, freqs, time_index):
    plt.figure(figsize=(10, 5))
    plt.contourf(time_index, freqs, coherence, 100, cmap='jet')
    plt.colorbar(label='Wavelet Coherence')
    plt.title('Wavelet Coherence between Sentiment and Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Days)')
    plt.show()

# Script Execution
if __name__ == "__main__":
    data = fetch_data('GME')
    data = create_features(data)

    scaler, lstm_model, X_test, y_test, predictions, mse = predictive_modeling(data[['Lag1', 'Lag2', 'Lag3']], data['Close'])
    print("MSE from LSTM Model: ", mse)

    visualize_predictions(scaler, X_test, y_test, predictions)

    # Assume 'perform_wavelet_coherence' function is defined elsewhere and returns 'coherence' and 'freqs'
    # coherence, freqs = perform_wavelet_coherence(data['Sentiment'], data['Close'])
    # plot_wavelet_coherence_heatmap(coherence, freqs, data.index)
