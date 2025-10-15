# File: train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
df = pd.read_csv('grid_data.csv')
X = df[['hour', 'node_id', 'load', 'temp', 'storm_active']]
y = df['failed_in_next_hour']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
with open('grid_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Created grid_model.pkl")