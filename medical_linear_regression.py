import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("../data/medical_data.csv")

X = data[['Age', 'BMI', 'BloodPressure']]
y = data['Glucose']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Model coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coefficients)

# Visualization (Glucose vs BMI)
plt.scatter(data['BMI'], data['Glucose'])
plt.plot(data['BMI'], model.predict(X), linestyle='--')
plt.xlabel("BMI")
plt.ylabel("Glucose Level")
plt.title("Glucose Prediction vs BMI")
plt.show()
