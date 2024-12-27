# Traffic-Volume-Prediction
This project implements a machine learning solution to predict traffic volume based on weather conditions and time-based features. Using a Random Forest model, we achieved 76.69% accuracy in predicting traffic patterns.
Traffic Volume Prediction

Overview
This project implements a machine learning solution to predict traffic volume based on weather conditions and time-based features. Using a Random Forest model, we achieved 76.69% accuracy in predicting traffic patterns.

Dataset Features
traffic_volume: Target variable - Number of vehicles
holiday: String indicating if the date is a holiday
temp: Temperature in Kelvin
rain_1h: Amount of rain in last hour (mm)
snow_1h: Amount of snow in last hour (mm)
clouds_all: Cloud coverage percentage
weather_main: Main weather category
weather_description: Detailed weather description
date_time: Date and time of measurement

Technical Implementation
Data Preprocessing

Missing Value Treatment
pythonCopy# Target variable
df['traffic_volume'] = df['traffic_volume'].fillna(df['traffic_volume'].mean())
# Weather measurements
df['temp'] = df['temp'].fillna(df['temp'].median())
df['rain_1h'] = df['rain_1h'].fillna(0)
df['snow_1h'] = df['snow_1h'].fillna(0)

Feature Engineering
Datetime decomposition (hour, day, month, day_of_week)
Categorical encoding using LabelEncoder
Feature standardization using StandardScaler

Feature Selection
Used mutual information regression to identify most important features:

Hour (1.0517)
Temperature (0.1967)
Day of Week (0.1645)
Day (0.1128)
Month (0.1022)

Model Performance
Linear Regression: R² = 0.1328 (13.28%)
Random Forest: R² = 0.7669 (76.69%)

Random Forest Parameters:
pythonCopyRandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

Required Dependencies
Pandas
numpy
scikit-learn
matplotlib
seaborn

Usage
Data Preparation

pythonCopy# Load and preprocess data
df = pd.read_csv("train_set_dirty.csv")
df_processed = prepare_features(df)

Model Training
pythonCopy# Train Random Forest model
rf_model = RandomForestRegressor(parameters)
rf_model.fit(X_train_scaled, y_train)

Prediction
pythonCopy# Make predictions
y_pred = rf_model.predict(X_test_scaled)
File Structure
Copy├── data/
│   ├── train_set_dirty.csv
│   └── test_set_nogt.csv
├── notebooks/
│   └── traffic_volume_prediction.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
├── requirements.txt
└── README.md

Results
Achieved 76.69% accuracy using Random Forest
Hour of day is the strongest predictor
Temperature and day of week are significant factors
Model captures non-linear relationships effectively

Future Improvements

Feature Engineering
Create interaction features
Add rush hour indicators

Model Optimization
Hyperparameter tuning using GridSearchCV
Experiment with other algorithms (XGBoost, LightGBM)

Cross-Validation
Implement k-fold cross-validation
Add time-based validation

Contributing
Feel free to open issues and pull requests for any improvements.

