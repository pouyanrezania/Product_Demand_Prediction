import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("demand.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
features = ['Product_ID', 'WeekOfYear']
target = 'Sales'
X = df[features]
y = df[target]
X = pd.get_dummies(X, columns=['Product_ID'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R²:", r2)
