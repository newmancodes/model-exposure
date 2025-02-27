import pandas as pd
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

# Exploratory Data Analysis
print("Data Overview:")
print(df.describe())
