import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # scikit-learn
from sklearn.model_selection import train_test_split

# 1. Simulate Historical Data (Replace with actual data)
np.random.seed(42)
n_months = 120
data = pd.DataFrame({
    'roi': np.random.normal(0.005, 0.05, n_months) + np.linspace(0, 0.02, n_months),
    'volatility': np.random.uniform(0.01, 0.03, n_months),
    'market_trend': np.random.normal(0, 1, n_months)
})


# 2. Prepare Features: Predict ROI next month based on last 3 months
def create_lags(df, lags=3):
    df_lagged = df.copy()
    for lag in range(1, lags + 1):
        df_lagged[f'roi_lag_{lag}'] = df_lagged['roi'].shift(lag)
    return df_lagged.dropna()


data_lagged = create_lags(data)
X = data_lagged.drop(['roi'], axis=1)
y = data_lagged['roi']

# 3. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# 4. Monte Carlo Simulation Function (One Month per Call)
def predict_next_month(model, current_data):
    """
    Takes current data, creates lags, and predicts next month's ROI.
    """
    latest_lags = create_lags(current_data.tail(4))  # Need at least 3+1 for lag
    prediction = model.predict(latest_lags.drop(['roi'], axis=1).tail(1))

    # Add noise for Monte Carlo sampling
    # The standard deviation of residuals can be added here
    sampled_prediction = np.random.normal(prediction[0], 0.02)
    return sampled_prediction


# --- Monte Carlo Loop Example ---
simulated_returns = []
current_market = data.copy()

for _ in range(12):  # Simulate 12 future months
    next_roi = predict_next_month(rf_model, current_market)
    simulated_returns.append(next_roi)

    # Update market behavior for next prediction
    new_row = pd.DataFrame({'roi': [next_roi], 'volatility': [0.02], 'market_trend': [0.1]})
    current_market = pd.concat([current_market, new_row], ignore_index=True)

print("Simulated Returns:", simulated_returns)
