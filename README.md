# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
The aim is to predict daily temperature, PM2.5 pollution level, and Energy using environmental sensor data and the Random Forest Algorithm. The dataset used is **`weather-station-eee-block_2024_07_13.csv`**, which contains various environmental sensor readings.



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm

1. Load and preprocess the weather sensor data.
2. Extract relevant environmental and time features.
3. Split the data into training and testing sets.
4. Train Random Forest models for temperature, PM2.5, and energy.
5. Predict the outputs and calculate MAE, RMSE, and R².
 

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: HEMAVARSHINI D
RegisterNumber:  212225220039
*/

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

df["hour"] = df["time"].dt.hour
df["dayofyear"] = df["time"].dt.dayofyear
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek

features = [
    "hum",
    "co2",
    "illumination",
    "pressure",
    "pm2_5",
    "pm10",
    "wind_direction_angle",
    "wind_speed",
    "wind_speed_level",
    "tsr",
    "hour",
    "dayofyear",
    "month",
    "dayofweek"
]

X = df[features]
X = X.fillna(X.mean())

y_temp = df["tem"].fillna(df["tem"].mean())
y_pollution = df["pm2_5"].fillna(df["pm2_5"].mean())
y_energy = df["tsr"].fillna(df["tsr"].mean())

split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train_temp = y_temp.iloc[:split]
y_test_temp = y_temp.iloc[split:]

y_train_pollution = y_pollution.iloc[:split]
y_test_pollution = y_pollution.iloc[split:]

y_train_energy = y_energy.iloc[:split]
y_test_energy = y_energy.iloc[split:]

temp_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

pollution_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

energy_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

temp_model.fit(X_train, y_train_temp)
pollution_model.fit(X_train, y_train_pollution)
energy_model.fit(X_train, y_train_energy)

temp_pred = temp_model.predict(X_test)
pollution_pred = pollution_model.predict(X_test)
energy_pred = energy_model.predict(X_test)

mae_temp = mean_absolute_error(y_test_temp, temp_pred)
rmse_temp = np.sqrt(mean_squared_error(y_test_temp, temp_pred))
r2_temp = r2_score(y_test_temp, temp_pred)

mae_pollution = mean_absolute_error(y_test_pollution, pollution_pred)
rmse_pollution = np.sqrt(mean_squared_error(y_test_pollution, pollution_pred))
r2_pollution = r2_score(y_test_pollution, pollution_pred)

mae_energy = mean_absolute_error(y_test_energy, energy_pred)
rmse_energy = np.sqrt(mean_squared_error(y_test_energy, energy_pred))
r2_energy = r2_score(y_test_energy, energy_pred)

print("Random Forest Results")
print("---------------------")

print("\nTemperature Prediction")
print("MAE  :", f"{mae_temp:.2f}")
print("RMSE :", f"{rmse_temp:.2f}")
print("R2   :", f"{r2_temp:.2f}")

print("\nPM2.5 Pollution Prediction")
print("MAE  :", f"{mae_pollution:.2f}")
print("RMSE :", f"{rmse_pollution:.2f}")
print("R2   :", f"{r2_pollution:.2f}")

print("\nEnergy Prediction")
print("MAE  :", f"{mae_energy:.2f}")
print("RMSE :", f"{rmse_energy:.2f}")
print("R2   :", f"{r2_energy:.2f}")
```

## Output:
<img width="414" height="443" alt="Screenshot 2026-08-19 115357" src="https://github.com/user-attachments/assets/defe2299-40e6-4089-939a-32d97def8c42" />


## Result:
Thus, the Random Forest Algorithm was successfully applied to analyze environmental sensor data for predicting daily temperature, PM2.5 pollution levels, and energy consumption.
