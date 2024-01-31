import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

bank_data = pd.read_csv("user_activity.csv")

# Add synthetic features
bank_data['Date'] = pd.to_datetime(bank_data['Date'])
bank_data = bank_data.sort_values(by='Date')
bank_data['days_since_last_purchase'] = (bank_data['Date'] - bank_data['Date'].shift(1)).dt.days.fillna(0)

# Encode categorical features
label_encoder = LabelEncoder()
bank_data['Category'] = label_encoder.fit_transform(bank_data['Category'])
bank_data['FavoriteCategory'] = label_encoder.transform(bank_data['FavoriteCategory'])
X = bank_data[['Category', 'Cost', 'BalanceBefore', 'days_since_last_purchase']]
y = bank_data['FavoriteCategory']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
rf_model.fit(X_train_scaled, y_train)

import pickle


model_pkl_file = "preference_prediction_model.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(rf_model, file)