import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import re
import joblib  # Import joblib

# --- 1. LOAD AND CLEAN THE CSV DATA ---
print("Loading and cleaning historical CSV data...")

# Load the CSV file
df = pd.read_csv('HistoricalQuotes.csv')  # Replace with your actual filename

# Data Cleaning Steps:
# 1. Remove any extra spaces from column names
df.columns = df.columns.str.strip()

# 2. Rename 'Close/Last' to 'Close' (case-insensitive check)
df.columns = df.columns.str.replace('close/last', 'Close', case=False)
df.columns = df.columns.str.replace('Close/Last', 'Close', case=False)

# 3. Convert date column to datetime if it exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    print("Date column set as index.")
elif 'DATE' in df.columns:
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    print("DATE column set as index.")

# 4. Ensure we have the required OHLCV columns
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV. Available columns: {list(df.columns)}")

# 5. CRITICAL: Clean numeric columns - remove $, commas, and convert to float
def clean_numeric_column(series):
    """Remove dollar signs, commas, and convert to float"""
    if series.dtype == 'object':  # If it's a string/object type
        # Remove dollar signs, commas, and any other non-numeric characters except decimal point
        series = series.str.replace('$', '', regex=False)
        series = series.str.replace(',', '', regex=False)
        series = series.str.replace(' ', '', regex=False)
        # Convert to numeric, forcing errors to NaN
        series = pd.to_numeric(series, errors='coerce')
    return series

# Clean all numeric columns
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = clean_numeric_column(df[col])
    print(f"Cleaned {col} column: {df[col].dtype}")

# 6. Remove any rows with missing values after cleaning
initial_count = len(df)
df = df.dropna()
final_count = len(df)
print(f"Removed {initial_count - final_count} rows with missing/invalid values.")
print(f"Final data shape: {df.shape}")

# --- 2. FEATURE ENGINEERING ---
print("Engineering features...")
# Create target variable: Next day's closing price
df['Target'] = df['Close'].shift(-1)
# Drop the last row which has a NaN for 'Target'
df = df.iloc[:-1]

# Create additional useful features
df['HL_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100.0  # Daily volatility %
df['OC_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100.0  # Daily change %

# Check for any NaN values created during feature engineering
if df.isnull().values.any():
    print("Warning: NaN values detected after feature engineering. Removing them...")
    df = df.dropna()
    print(f"New data shape: {df.shape}")

# Define the feature set (X) and target (y)
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'HL_Pct', 'OC_Pct']
X = df[feature_columns].values
y = df['Target'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# --- 3. SCALE THE DATA (CRITICAL STEP) ---
print("Scaling data...")
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# --- 4. CREATE SEQUENCES FOR LSTM ---
lookback_days = 60  # How many past days to use for prediction

# Verify we have enough data
if len(X_scaled) < lookback_days + 1:
    raise ValueError(f"Not enough data. Need at least {lookback_days + 1} days, but only have {len(X_scaled)}")

X_data = []
y_data = []

for i in range(lookback_days, len(X_scaled)):
    X_data.append(X_scaled[i - lookback_days:i])  # Sequence of 60 days
    y_data.append(y_scaled[i, 0])  # The target value for that sequence

X_data, y_data = np.array(X_data), np.array(y_data)

print(f"Final LSTM dataset shape: {X_data.shape}")

# --- 5. BUILD THE LSTM MODEL ---
print("Building LSTM model...")
model = Sequential()
# First LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_data.shape[1], X_data.shape[2])))
model.add(Dropout(0.2))
# Second LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
# Dense layers
model.add(Dense(units=25))
model.add(Dense(units=1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')

# --- 6. TRAIN THE MODEL ---
print("Training the model... (This may take a few minutes)")
history = model.fit(X_data, y_data,
                    batch_size=32,
                    epochs=10,
                    validation_split=0.1,
                    verbose=1,
                    shuffle=False)

# --- 7. MAKE A PREDICTION FOR THE NEXT TRADING DAY ---
# Get the last 'lookback_days' of data
last_60_days = X_scaled[-lookback_days:]
last_60_days_reshaped = last_60_days.reshape(1, lookback_days, len(feature_columns))

# Predict the scaled next day price
predicted_price_scaled = model.predict(last_60_days_reshaped)

# --- INVERSE THE SCALING TO GET THE REAL PRICE ---
fake_row = np.zeros((1, len(feature_columns)))
fake_row[:, 3] = predicted_price_scaled[0, 0]  # 'Close' is at index 3
fake_row_inversed = scaler_x.inverse_transform(fake_row)
next_day_predicted_price = fake_row_inversed[0, 3]

print(f"\nPredicted Closing Price for the Next Trading Day: ${next_day_predicted_price:.2f}")
print(f"Last Actual Closing Price: ${df['Close'].iloc[-1]:.2f}")

# --- 8. GENERATE A TRADING SIGNAL ---
current_price = df['Close'].iloc[-1]
signal = "HOLD"
reason = "No significant price movement predicted."

if next_day_predicted_price > current_price * 1.005:  # > 0.5% increase
    signal = "BUY"
    reason = f"Predicted price increase of {(next_day_predicted_price/current_price - 1)*100:.2f}%."
elif next_day_predicted_price < current_price * 0.995:  # > 0.5% decrease
    signal = "SELL"
    reason = f"Predicted price decrease of {(1 - next_day_predicted_price/current_price)*100:.2f}%."

print(f"\nTRADING SIGNAL: {signal}")
print(f"REASON: {reason}")

# --- 9. PLOT TRAINING LOSS ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progress')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# --- 10. SAVE THE MODEL AND SCALERS USING JOBLIB ---
print("Saving model and scalers using joblib...")

# Save the Keras model
model.save('aapl_price_prediction_model.h5')

# Save the scalers using joblib (more efficient for scikit-learn objects)
joblib.dump(scaler_x, 'scaler_x.joblib')
joblib.dump(scaler_y, 'scaler_y.joblib')

# Also save the feature columns list for reference in Flask app
joblib.dump(feature_columns, 'feature_columns.joblib')
joblib.dump(lookback_days, 'lookback_days.joblib')

print("\nModel and scalers saved successfully using joblib!")
print("Files created:")
print("- 'aapl_price_prediction_model.h5' (Keras model)")
print("- 'scaler_x.joblib' (Feature scaler)")
print("- 'scaler_y.joblib' (Target scaler)")
print("- 'feature_columns.joblib' (Feature names)")
print("- 'lookback_days.joblib' (Lookback window)")

# --- BONUS: Check data types to verify cleaning worked ---
print("\nFinal data types:")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].dtypes)