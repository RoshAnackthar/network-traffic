import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
@st.cache_data
def load_data():
    train_data = pd.read_csv('UNSW_NB15_training-set.csv')
    test_data = pd.read_csv('UNSW_NB15_testing-set.csv')
    return train_data, test_data

train_data, test_data = load_data()

# Sidebar input to select features and parameters
st.sidebar.header("Select Features")

# Sample feature selection for this app (you can change based on your data)
features = ['ct_dst_sport_ltm', 'ct_dst_src_ltm', 'sbytes', 'dbytes']

# Select the features
X_train = train_data[features]
y_train = train_data['Label']
X_test = test_data[features]
y_test = test_data['Label']

# Train Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Display results
st.subheader("Mean Squared Error")
st.write(mse)

st.subheader("Sample Predictions")
results = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred)})
st.write(results.head())