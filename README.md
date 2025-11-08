# Natural_gas_forecasting_with_ML

Quantitative Natural Gas Forecasting using Machine Learning

This project models and forecasts natural gas prices to enable the pricing of storage contracts for a commodity trading desk.
It combines quantitative finance, time-series modeling, and machine learning to extrapolate market data, capture seasonal trends, and generate forward-looking price estimates.

🧠 Background

Commodity storage contracts allow firms to buy and store physical commodities (like natural gas) during low-demand periods and sell them during high-demand seasons.
Accurate pricing of such contracts depends on estimating the expected future prices of natural gas at any injection or withdrawal date.

However, available market data is often sparse or noisy, requiring advanced methods to interpolate and extrapolate prices across time.

This project was completed as part of a J.P. Morgan quantitative research work experience simulation.

🎯 Objectives

Improve data granularity from external market feeds.

Model seasonal price behavior across months and years.

Forecast future natural gas prices using machine learning.

Evaluate model accuracy and visualize trends for pricing applications.

🧩 Approach
1. Data Preparation

Collected historical natural gas price data (e.g., Henry Hub, NBP).

Cleaned, interpolated missing values, and adjusted for seasonal cycles.

Engineered calendar-based features (month, season, day-of-year).

2. Exploratory Data Analysis

Visualized seasonal patterns, price volatility, and yearly trends.

Decomposed time series into trend, seasonal, and residual components.

3. Modeling & Forecasting

Several models were benchmarked:

ARIMA / SARIMA: Classical seasonal time-series baselines.

Prophet: Seasonality-aware additive model for interpretable forecasting.

LSTM Neural Network: Captures nonlinear temporal dependencies.

Gradient Boosting (XGBoost / LightGBM): Uses engineered features for regression-based forecasting.

4. Evaluation

Compared models using MAE, RMSE, and MAPE.

Cross-validated predictions across years and seasons.

Generated price extrapolations for future months.

📈 Results

The machine learning models (XGBoost / LSTM) outperformed traditional ARIMA in capturing non-linear seasonal behavior.

Forecasts show a clear annual winter price premium, consistent with demand spikes.

The resulting model provides smooth extrapolation suitable for natural gas storage contract pricing.

🔧 Technologies Used
Category	Tools
Language	Python
Data	Pandas, NumPy,	Matplotlib, Seaborn
Modeling	Scikit-learn, Keras, GitHub
