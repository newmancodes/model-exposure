import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory for saving plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('data/housing.csv')
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0}.get)
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0}.get)
df['basement'] = df['basement'].map({'yes': 1, 'no': 0}.get)
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0}.get)
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0}.get)
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0}.get)
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}.get)

# Exploratory Data Analysis
print("Data Overview:")
print(df.describe())

# Visualize relationships
plt.figure(figsize=(24, 16))

plt.subplot(2, 2, 1)
sns.scatterplot(x='area', y='price', data=df)
plt.title('Price vs Area')

plt.subplot(2, 2, 2)
sns.scatterplot(x='bedrooms', y='price', data=df)
plt.title('Price vs Bedrooms')

plt.subplot(2, 2, 3)
sns.scatterplot(x='bathrooms', y='price', data=df)
plt.title('Price vs Bathrooms')

plt.subplot(2, 2, 4)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.savefig('plots/exploratory_analysis.png', dpi=300)
plt.close()  # Close the figure to free memory

print("Saved exploratory analysis plots to 'plots/exploratory_analysis.png'")

# Prepare data for modeling
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
y = df['price']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Create a directory for saving the model if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
pickle.dump(model, open('models/linear_regression.pkl', 'wb'))

# Display model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
# Feature importance visualization

plt.figure(figsize=(10, 6))
features = X.columns
importance = np.abs(model.coef_)
features_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
features_importance = features_importance.sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=features_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300)
plt.close()  # Close the figure to free memory

print("Saved feature importance plot to 'plots/feature_importance.png'")

# Create a plot for actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=300)
plt.close()

print("Saved actual vs predicted plot to 'plots/actual_vs_predicted.png'")

# Save the residuals plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('plots/residuals.png', dpi=300)
plt.close()

print("Saved residuals plot to 'plots/residuals.png'")